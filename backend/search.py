"""Pure semantic search over image descriptions using Zvec and SentenceTransformers."""

import threading
from pathlib import Path
import zvec
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

from backend.config import ZVEC_DIR, ZVEC_DIMENSION, EMBEDDING_MODEL, SEARCH_TOP_K
from backend.synonyms import get_entity_synonyms, expand_query_text


def calibrate_score(raw_score: float) -> float:
    """
    Continuous calibration of cosine similarity to intuitive UI percentages.
    Spans [0.18, 0.58] smoothly without flat clipping ceilings.
    """
    if raw_score <= 0.18:
        return 0.0
    val = (raw_score - 0.18) / (0.58 - 0.18)
    val = max(0.0, min(1.0, val))
    return round(0.50 + 0.48 * (val ** 0.8), 4)


RELATIONAL_CONNECTORS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "of", "and", "or", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "photo", "photos", "image", "images", "picture", "pictures", "showing",
    "there", "this", "that", "these", "those", "near", "next", "behind",
    "front", "holding", "standing", "sitting", "lying", "laying", "beside",
    "under", "over", "along", "across"
}


def token_satisfaction(sim_val: float) -> float:
    """
    Continuous transfer function mapping token cosine similarity to query entity satisfaction.
    - >= 0.78: 1.0 (exact match or direct synonym, e.g. noodles for pasta, sea for ocean)
    - 0.62 - 0.78: smooth transition for close hyponyms
    - <= 0.62: 0.0 (distractors and sibling categories, e.g. pizza vs pasta @ 0.617, river vs ocean @ 0.509)
    """
    if sim_val <= 0.62:
        return 0.0
    if sim_val >= 0.78:
        return 1.0
    return (sim_val - 0.62) / (0.78 - 0.62)


def split_sentences(text: str) -> list[str]:
    import re
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]
    return sents if sents else [text]


# ── Zvec collection schema ──────────────────────────────

_ZVEC_SCHEMA = zvec.CollectionSchema(
    name="image_descriptions",
    fields=[
        zvec.FieldSchema(
            name="image_id",
            data_type=zvec.DataType.INT64,
        ),
        zvec.FieldSchema(
            name="document",
            data_type=zvec.DataType.STRING,
        ),
    ],
    vectors=[
        zvec.VectorSchema(
            name="embedding",
            data_type=zvec.DataType.VECTOR_FP32,
            dimension=ZVEC_DIMENSION,
            index_param=zvec.HnswIndexParam(
                metric_type=zvec.MetricType.COSINE,
            ),
        ),
    ],
)


class SemanticSearch:
    """Manages Zvec vector collection and natural language semantic query embedding (Process Singleton)."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        with self._lock:
            if getattr(self, "_initialized", False):
                return
            ZVEC_DIR.mkdir(parents=True, exist_ok=True)
            collection_dir = ZVEC_DIR / "image_descriptions"
            collection_path = str(collection_dir)

            if collection_dir.exists():
                lock_file = collection_dir / "LOCK"
                lock_file.touch(exist_ok=True)
                self._collection = zvec.open(path=collection_path)
            else:
                self._collection = zvec.create_and_open(
                    path=collection_path,
                    schema=_ZVEC_SCHEMA,
                )
            self._embedder = None
            self._initialized = True

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            is_local = Path(EMBEDDING_MODEL).exists()
            model_source = "bundled offline package" if is_local else "Hugging Face"
            print(f"Loading embedding model from {model_source}: {EMBEDDING_MODEL}...")
            try:
                self._embedder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as e:
                if is_local:
                    print(f"Bundled model load error ({e}), trying default model name...")
                    self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                else:
                    raise
        return self._embedder

    def add(self, image_id: int, enriched_text: str) -> None:
        """Add or update a single image's embedding."""
        doc_id = str(image_id)
        embedding = self.embedder.encode(enriched_text).tolist()
        with self._lock:
            self._collection.upsert(
                zvec.Doc(
                    id=doc_id,
                    vectors={"embedding": embedding},
                    fields={"image_id": image_id, "document": enriched_text},
                )
            )

    def add_batch(self, items: list[tuple[int, str]]) -> None:
        """Add multiple (image_id, enriched_text) pairs."""
        if not items:
            return
        texts = [t for _, t in items]
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

        docs = []
        for i, (iid, text) in enumerate(items):
            docs.append(
                zvec.Doc(
                    id=str(iid),
                    vectors={"embedding": embeddings[i]},
                    fields={"image_id": iid, "document": text},
                )
            )
        with self._lock:
            self._collection.upsert(docs)
            self._collection.optimize()

    def delete(self, image_ids: list[int]) -> None:
        """Delete documents by image IDs."""
        if not image_ids:
            return
        str_ids = [str(i) for i in image_ids]
        try:
            with self._lock:
                self._collection.delete(ids=str_ids)
        except Exception as e:
            print(f"Zvec delete error: {e}")

    def reset(self) -> None:
        """Delete and recreate the entire collection."""
        import shutil
        collection_path = str(ZVEC_DIR / "image_descriptions")
        try:
            self._collection = None
        except Exception:
            pass
        # Remove the collection directory
        coll_path = ZVEC_DIR / "image_descriptions"
        if coll_path.exists():
            shutil.rmtree(str(coll_path), ignore_errors=True)
        # Recreate fresh
        self._collection = zvec.create_and_open(
            path=collection_path,
            schema=_ZVEC_SCHEMA,
        )

    def query(self, text: str, top_k: int = SEARCH_TOP_K, filter_mode: str = "balanced") -> list[dict]:
        """
        Two-stage retrieval pipeline:
        Stage 1: Fast Zvec HNSW vector retrieval over 384-d semantic embedding space.
        Stage 2: Subject/Object Semantic Intersection Filtering (strict intersection gate 
                 requiring all query entities, rather than union/partial blending) + multi-scale ranking.
        """
        import math
        query_text = text.strip()
        if not query_text:
            return []

        # Extract substantive entities (subjects and objects)
        q_clean = query_text.strip()
        q_words = [w.strip(".,;:?!'\"()[]{}").lower() for w in q_clean.split()]
        entities = [w for w in q_words if w and w not in RELATIONAL_CONNECTORS]
        if not entities:
            entities = q_words

        q_emb = self.embedder.encode(q_clean, convert_to_tensor=True, normalize_embeddings=True)
        embedding = q_emb.tolist()

        # Generate synonym expansion if applicable
        expanded_query = expand_query_text(q_clean)
        exp_emb = None
        if expanded_query != q_clean:
            exp_emb = self.embedder.encode(expanded_query, convert_to_tensor=True, normalize_embeddings=True)

        # ── Stage 1: Zvec ANN Candidate Retrieval ──
        total_in_db = self.count or 1
        fetch_k = min(max(top_k * 3, 50), total_in_db)

        raw_results = self._collection.query(
            queries=zvec.Query(
                field_name="embedding",
                vector=embedding,
            ),
            topk=fetch_k,
        ) or []

        # If synonyms exist, fetch candidates for the expanded query as well
        if exp_emb is not None:
            exp_results = self._collection.query(
                queries=zvec.Query(
                    field_name="embedding",
                    vector=exp_emb.tolist(),
                ),
                topk=fetch_k,
            ) or []
        else:
            exp_results = []

        # Deduplicate candidates by image_id, preserving best raw similarity
        candidates_by_id = {}
        for doc in list(raw_results) + list(exp_results):
            raw_sim = max(0.0, 1.0 - float(doc.score))
            doc_text = doc.fields.get("document", "") if doc.fields else ""
            image_id = doc.fields.get("image_id", 0) if doc.fields else 0
            if not image_id:
                try:
                    image_id = int(doc.id)
                except (ValueError, TypeError):
                    continue

            if image_id not in candidates_by_id or raw_sim > candidates_by_id[image_id]["raw_score"]:
                candidates_by_id[image_id] = {
                    "image_id": image_id,
                    "raw_score": raw_sim,
                    "doc_sim": raw_sim,
                    "enriched_text": doc_text,
                    "sents": split_sentences(doc_text),
                }

        candidates = list(candidates_by_id.values())
        if not candidates:
            return []

        # Build entity synonym map for Stage 2
        entity_syn_map = {e: [e] + get_entity_synonyms(e) for e in entities}
        encoded_syn_map = {
            e: self.embedder.encode(syns, convert_to_tensor=True, normalize_embeddings=True)
            for e, syns in entity_syn_map.items()
        }

        # ── Stage 2: Subject/Object Semantic Intersection Filtering & Ranking ──
        survived = []
        for c in candidates:
            # Multi-scale sentence-level dense similarity
            s_embs = self.embedder.encode(c["sents"], convert_to_tensor=True, normalize_embeddings=True)
            sent_sims = F.cosine_similarity(q_emb.unsqueeze(0), s_embs)
            max_sent_sim = sent_sims.max().item()

            if exp_emb is not None:
                exp_sent_sims = F.cosine_similarity(exp_emb.unsqueeze(0), s_embs)
                max_sent_sim = max(max_sent_sim, 0.95 * exp_sent_sims.max().item())

            base_dense = 0.50 * c["doc_sim"] + 0.50 * max_sent_sim

            # Candidate words
            all_words = [w.strip(".,;:?!'\"()[]{}").lower() for w in c["enriched_text"].split()]
            all_words = [w for w in all_words if len(w) > 1]
            if not all_words:
                continue

            all_w_embs = self.embedder.encode(all_words, convert_to_tensor=True, normalize_embeddings=True)

            # Evaluate each required entity across document words (with synonym bridge)
            doc_sats = []
            for e, s_embs_tensor in encoded_syn_map.items():
                t_sims = F.cosine_similarity(s_embs_tensor.unsqueeze(1), all_w_embs.unsqueeze(0), dim=2)
                aspect_max = t_sims.max(dim=1).values
                direct_sat = token_satisfaction(aspect_max[0].item())
                syn_sat = max([token_satisfaction(x.item()) for x in aspect_max[1:]]) if len(aspect_max) > 1 else 0.0
                best_entity_sat = max(direct_sat, 0.92 * syn_sat)
                doc_sats.append(best_entity_sat)

            # ── Entity Presence & Intersection Constraint ──
            # Required query entities must be semantically satisfied in the candidate document.
            min_doc_sat = min(doc_sats)
            if min_doc_sat <= 0.05:
                # Fails entity presence filter
                continue

            # Sentence-level co-occurrence (sharpest local conjunction)
            best_sent_sat = 0.0
            for sent in c["sents"]:
                sent_words = [w.strip(".,;:?!'\"()[]{}").lower() for w in sent.split()]
                sent_words = [w for w in sent_words if len(w) > 1]
                if not sent_words:
                    continue
                w_embs = self.embedder.encode(sent_words, convert_to_tensor=True, normalize_embeddings=True)

                sent_entity_sats = []
                for e, s_embs_tensor in encoded_syn_map.items():
                    st_sims = F.cosine_similarity(s_embs_tensor.unsqueeze(1), w_embs.unsqueeze(0), dim=2)
                    st_max = st_sims.max(dim=1).values
                    st_direct = token_satisfaction(st_max[0].item())
                    st_syn = max([token_satisfaction(x.item()) for x in st_max[1:]]) if len(st_max) > 1 else 0.0
                    sent_entity_sats.append(max(st_direct, 0.92 * st_syn))

                s_min = min(sent_entity_sats)
                s_geom = math.prod(sent_entity_sats) ** (1.0 / len(sent_entity_sats))
                sent_score = 0.50 * s_min + 0.50 * s_geom
                if sent_score > best_sent_sat:
                    best_sent_sat = sent_score

            doc_geom = math.prod(doc_sats) ** (1.0 / len(doc_sats))
            doc_conj = 0.60 * min_doc_sat + 0.40 * doc_geom

            intersection_score = max(best_sent_sat, 0.70 * doc_conj)

            final_raw = base_dense * intersection_score
            c["raw_score"] = round(final_raw, 4)
            c["score"] = calibrate_score(c["raw_score"])
            survived.append(c)

        survived.sort(key=lambda x: x["raw_score"], reverse=True)

        if filter_mode == "all" or not survived:
            return survived[:top_k]

        # ── Stage 3: Dynamic Topological Elbow Cutoff ──
        min_noise_floor = 0.18 if filter_mode != "broad" else 0.14
        drop_threshold = 0.080 if filter_mode != "broad" else 0.10

        filtered = []
        if survived and survived[0]["raw_score"] >= min_noise_floor:
            filtered.append(survived[0])
            for i in range(1, len(survived)):
                curr = survived[i]
                prev = survived[i - 1]
                if curr["raw_score"] < min_noise_floor:
                    break
                drop = prev["raw_score"] - curr["raw_score"]
                if drop >= drop_threshold:
                    break
                if curr["raw_score"] < survived[0]["raw_score"] * 0.45:
                    break
                filtered.append(curr)

        return filtered[:top_k]

        return filtered[:top_k]

    @property
    def count(self) -> int:
        try:
            stats = self._collection.stats
            if hasattr(stats, 'doc_count'):
                return int(stats.doc_count)
            if hasattr(stats, 'total_doc_count'):
                return int(stats.total_doc_count)
            if isinstance(stats, dict):
                return int(stats.get('doc_count', stats.get('total_doc_count', 0)))
            return 0
        except Exception:
            return 0
