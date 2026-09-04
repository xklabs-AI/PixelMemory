"""Pure semantic search over image descriptions using ChromaDB and SentenceTransformers."""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

from backend.config import CHROMA_DIR, EMBEDDING_MODEL, SEARCH_TOP_K


def calibrate_score(raw_score: float) -> float:
    """
    Continuous calibration of cosine similarity to intuitive UI percentages.
    Spans [0.20, 0.65] smoothly without flat clipping ceilings.
    """
    if raw_score <= 0.20:
        return 0.0
    val = (raw_score - 0.20) / (0.65 - 0.20)
    val = max(0.0, min(1.0, val))
    return round(0.50 + 0.48 * (val ** 0.8), 4)


STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "of", "and", "or", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "photo", "photos", "image", "images", "picture", "pictures"
}


def token_satisfaction(sim_val: float) -> float:
    """
    Continuous transfer function mapping token cosine similarity to query facet satisfaction.
    - >= 0.88: 1.0 (exact or near-exact word)
    - 0.76 - 0.87: 0.60 - 0.95 (strong synonym e.g. kitten/cat)
    - 0.65 - 0.75: 0.05 - 0.35 (contrast / co-hyponym e.g. blue/red)
    - <= 0.60: 0.0 (unrelated)
    """
    if sim_val <= 0.60:
        return 0.0
    val = (sim_val - 0.60) / (0.88 - 0.60)
    val = min(1.0, max(0.0, val))
    return val ** 1.6


def split_sentences(text: str) -> list[str]:
    import re
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]
    return sents if sents else [text]


class SemanticSearch:
    """Manages ChromaDB vector collection and natural language semantic query embedding."""

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="image_descriptions",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL}...")
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def add(self, image_id: int, enriched_text: str) -> None:
        """Add or update a single image's embedding."""
        doc_id = str(image_id)
        embedding = self.embedder.encode(enriched_text).tolist()
        self._collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[enriched_text],
            metadatas=[{"image_id": image_id}],
        )

    def add_batch(self, items: list[tuple[int, str]]) -> None:
        """Add multiple (image_id, enriched_text) pairs."""
        if not items:
            return
        ids = [str(iid) for iid, _ in items]
        texts = [t for _, t in items]
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()
        metadatas = [{"image_id": iid} for iid, _ in items]
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas,
        )

    def query(self, text: str, top_k: int = SEARCH_TOP_K, filter_mode: str = "balanced") -> list[dict]:
        """
        Multi-granularity semantic search with neural sentence-level aspect conjunction
        and dynamic topological elbow cutoff.
        
        Zero hardcoded keywords:
        1. Multi-scale dense alignment (whole document + sharpest local sentence).
        2. Sentence-level soft-AND conjunction ensuring multi-facet queries (e.g. 'red sky')
           require co-occurrence with real semantic satisfaction, eliminating attribute mismatches.
        3. Dynamic elbow gap filtering cutting off step-drops into background noise.
        """
        import math
        query_text = text.strip()
        if not query_text:
            return []

        # 1. First-stage retrieval: dense vector similarity over document collection
        q_clean = query_text.strip()
        q_words = [w.strip(".,;:?!'\"()[]{}").lower() for w in q_clean.split()]
        substantive = [w for w in q_words if w and w not in STOP_WORDS]

        q_emb = self.embedder.encode(q_clean, convert_to_tensor=True, normalize_embeddings=True)
        embedding = q_emb.tolist()

        total_in_db = self._collection.count() or 1
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k * 2, total_in_db),
            include=["documents", "distances", "metadatas"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        candidates = []
        for i, doc_id in enumerate(results["ids"][0]):
            raw_sim = 1.0 - results["distances"][0][i]
            doc_text = results["documents"][0][i]
            sents = split_sentences(doc_text)
            candidates.append({
                "image_id": results["metadatas"][0][i]["image_id"],
                "raw_score": raw_sim,
                "doc_sim": raw_sim,
                "enriched_text": doc_text,
                "sents": sents,
            })

        # Pre-encode substantive query facets for multi-term queries
        sub_embs = None
        if len(substantive) > 1:
            sub_embs = self.embedder.encode(substantive, convert_to_tensor=True, normalize_embeddings=True)

        # 2. Multi-granularity blending & sentence-level aspect conjunction
        for c in candidates:
            # Sentence-level max dense score
            s_embs = self.embedder.encode(c["sents"], convert_to_tensor=True, normalize_embeddings=True)
            sent_sims = F.cosine_similarity(q_emb.unsqueeze(0), s_embs)
            max_sent_sim = sent_sims.max().item()

            base_dense = 0.50 * c["doc_sim"] + 0.50 * max_sent_sim

            conjunction_mult = 1.0
            if sub_embs is not None:
                best_sent_conjunction = 0.0

                for sent in c["sents"]:
                    sent_words = [w.strip(".,;:?!'\"()[]{}").lower() for w in sent.split()]
                    sent_words = [w for w in sent_words if len(w) > 1]
                    if not sent_words:
                        continue
                    w_embs = self.embedder.encode(sent_words, convert_to_tensor=True, normalize_embeddings=True)
                    t_sims = F.cosine_similarity(sub_embs.unsqueeze(1), w_embs.unsqueeze(0), dim=2)
                    aspect_max = t_sims.max(dim=1).values
                    aspect_sats = [token_satisfaction(s.item()) for s in aspect_max]

                    sent_min = min(aspect_sats)
                    sent_geom = math.prod(aspect_sats) ** (1.0 / len(aspect_sats))
                    sent_score = 0.50 * sent_min + 0.50 * sent_geom
                    if sent_score > best_sent_conjunction:
                        best_sent_conjunction = sent_score

                # Document-wide fallback for loosely coupled facets
                all_words = [w.strip(".,;:?!'\"()[]{}").lower() for w in c["enriched_text"].split()]
                all_words = [w for w in all_words if len(w) > 1]
                if all_words:
                    all_w_embs = self.embedder.encode(all_words, convert_to_tensor=True, normalize_embeddings=True)
                    doc_t_sims = F.cosine_similarity(sub_embs.unsqueeze(1), all_w_embs.unsqueeze(0), dim=2)
                    doc_aspect_max = doc_t_sims.max(dim=1).values
                    doc_sats = [token_satisfaction(s.item()) for s in doc_aspect_max]
                    doc_geom = math.prod(doc_sats) ** (1.0 / len(doc_sats))
                    doc_min = min(doc_sats)
                    doc_fallback = 0.50 * doc_min + 0.50 * doc_geom
                else:
                    doc_fallback = 0.0

                conjunction_mult = max(best_sent_conjunction, 0.40 * doc_fallback)

            final_raw = base_dense * conjunction_mult
            c["raw_score"] = round(final_raw, 4)
            c["score"] = calibrate_score(c["raw_score"])

        candidates.sort(key=lambda x: x["raw_score"], reverse=True)

        if filter_mode == "all" or not candidates:
            return candidates[:top_k]

        # 3. Dynamic Topological Elbow Cutoff
        # Filters out step drops into background noise while preserving dense categorical clusters
        min_noise_floor = 0.22 if filter_mode != "broad" else 0.16
        drop_threshold = 0.070 if filter_mode != "broad" else 0.095

        filtered = []
        if candidates and candidates[0]["raw_score"] >= min_noise_floor:
            filtered.append(candidates[0])
            for i in range(1, len(candidates)):
                curr = candidates[i]
                prev = candidates[i - 1]
                if curr["raw_score"] < min_noise_floor:
                    break
                drop = prev["raw_score"] - curr["raw_score"]
                if drop >= drop_threshold:
                    break
                if curr["raw_score"] < candidates[0]["raw_score"] * 0.48:
                    break
                filtered.append(curr)

        return filtered[:top_k]

    @property
    def count(self) -> int:
        return self._collection.count()
