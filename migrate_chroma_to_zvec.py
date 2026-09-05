"""
One-time migration: transfer existing embeddings from ChromaDB to Zvec.

No re-encoding needed — existing embedding vectors are copied directly.

Usage:
    python migrate_chroma_to_zvec.py
"""

import sys
from pathlib import Path

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent))

from backend.config import CHROMA_DIR, ZVEC_DIR, ZVEC_DIMENSION


def migrate():
    chroma_path = CHROMA_DIR
    if not chroma_path.exists():
        print(f"No ChromaDB directory found at {chroma_path}")
        print("Nothing to migrate — zvec collection will be created fresh on first use.")
        return

    print(f"ChromaDB directory found at: {chroma_path}")

    # Import ChromaDB (must still be installed for migration)
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("\nERROR: chromadb package is required for migration.")
        print("Install temporarily with: pip install chromadb")
        print("You can uninstall it after migration: pip uninstall chromadb")
        return

    import zvec

    # 1. Read all data from ChromaDB
    print("\n[1/4] Reading existing ChromaDB collection...")
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection("image_descriptions")
    except Exception as e:
        print(f"  Could not find 'image_descriptions' collection: {e}")
        print("  Nothing to migrate.")
        return

    total = collection.count()
    print(f"  Found {total} documents in ChromaDB")

    if total == 0:
        print("  Empty collection — nothing to migrate.")
        return

    # Fetch all data in batches
    batch_size = 500
    all_docs = []
    for offset in range(0, total, batch_size):
        batch = collection.get(
            include=["embeddings", "documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        for i, doc_id in enumerate(batch["ids"]):
            all_docs.append({
                "id": doc_id,
                "embedding": batch["embeddings"][i],
                "document": batch["documents"][i] if batch["documents"] else "",
                "image_id": batch["metadatas"][i].get("image_id", 0) if batch["metadatas"] else 0,
            })

    print(f"  Read {len(all_docs)} documents with embeddings")

    # 2. Detect embedding dimension from first document
    dim = len(all_docs[0]["embedding"])
    print(f"\n[2/4] Detected embedding dimension: {dim}")
    if dim != ZVEC_DIMENSION:
        print(f"  WARNING: Expected dimension {ZVEC_DIMENSION}, got {dim}")
        print(f"  Using detected dimension {dim} for zvec schema")

    # 3. Create zvec collection
    print(f"\n[3/4] Creating zvec collection at {ZVEC_DIR / 'image_descriptions'}...")
    ZVEC_DIR.mkdir(parents=True, exist_ok=True)
    zvec_path = str(ZVEC_DIR / "image_descriptions")

    schema = zvec.CollectionSchema(
        name="image_descriptions",
        fields=[
            zvec.FieldSchema(name="image_id", data_type=zvec.DataType.INT64),
            zvec.FieldSchema(name="document", data_type=zvec.DataType.STRING),
        ],
        vectors=[
            zvec.VectorSchema(
                name="embedding",
                data_type=zvec.DataType.VECTOR_FP32,
                dimension=dim,
                index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
            ),
        ],
    )

    # Remove existing zvec collection if it exists (clean migration)
    import shutil
    zvec_coll_path = ZVEC_DIR / "image_descriptions"
    if zvec_coll_path.exists():
        print(f"  Removing existing zvec collection at {zvec_coll_path}...")
        shutil.rmtree(str(zvec_coll_path))

    zvec_collection = zvec.create_and_open(path=zvec_path, schema=schema)

    # 4. Batch insert into zvec
    print(f"\n[4/4] Migrating {len(all_docs)} documents to zvec...")
    batch_size = 256
    migrated = 0

    for batch_start in range(0, len(all_docs), batch_size):
        batch = all_docs[batch_start : batch_start + batch_size]
        zvec_docs = []
        for d in batch:
            image_id = d["image_id"]
            if not image_id:
                try:
                    image_id = int(d["id"])
                except (ValueError, TypeError):
                    image_id = 0
            zvec_docs.append(
                zvec.Doc(
                    id=d["id"],
                    vectors={"embedding": d["embedding"]},
                    fields={"image_id": image_id, "document": d["document"] or ""},
                )
            )
        zvec_collection.upsert(zvec_docs)
        migrated += len(batch)
        pct = (migrated / len(all_docs)) * 100
        print(f"  {migrated}/{len(all_docs)} ({pct:.0f}%)")

    # Optimize index for peak search performance
    print("\n  Optimizing zvec HNSW index...")
    zvec_collection.optimize()

    print(f"\n{'=' * 60}")
    print(f"Migration complete!")
    print(f"  Migrated: {migrated} documents")
    print(f"  Source:   {chroma_path}")
    print(f"  Target:   {ZVEC_DIR / 'image_descriptions'}")
    print(f"\nThe old ChromaDB directory at {chroma_path} is no longer needed.")
    print(f"You can delete it manually when ready:")
    print(f"  rmdir /s /q \"{chroma_path}\"")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    migrate()
