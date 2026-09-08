"""Recursively scan directories for images, hash and deduplicate, sync additions/removals."""

import os
from pathlib import Path

import xxhash
from tqdm import tqdm

from backend.config import SUPPORTED_EXTENSIONS, HASH_CHUNK_SIZE
from backend.db import get_conn, upsert_image, delete_images


def hash_file(path: str) -> str:
    """Fast content hash using xxHash (non-cryptographic, very fast)."""
    h = xxhash.xxh64()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _normalize_path(p: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def scan_directory(
    root: str | Path,
    rescan_mode: str = "incremental",
    remove_deleted: bool = True,
    callback=None
) -> dict:
    """
    Walk a directory tree, find images, hash them, insert into DB.
    Supports:
      - rescan_mode: 'incremental' (leave existing untouched, only add new) or 'full' (re-process existing)
      - remove_deleted: cleans up database records, thumbnails, and search vectors for photos deleted from disk.
    Returns stats: {found, new, duplicate, deleted_pruned, reprocessed, errors}.
    Optional callback(current, total, file_name) for live progress tracking.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    root_norm = _normalize_path(root)

    # Collect all image paths currently on disk
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.normpath(os.path.join(dirpath, fname)))

    disk_paths_set = {_normalize_path(p) for p in image_paths}

    stats = {
        "found": len(image_paths),
        "new": 0,
        "duplicate": 0,
        "deleted_pruned": 0,
        "reprocessed": 0,
        "errors": 0,
    }

    with get_conn() as conn:
        all_db_rows = conn.execute("SELECT id, file_path, file_hash FROM images").fetchall()

        # Identify rows that belong to this folder tree
        folder_db_rows = []
        for r in all_db_rows:
            r_norm = _normalize_path(r["file_path"])
            if r_norm == root_norm or r_norm.startswith(root_norm + os.sep) or r_norm.startswith(root_norm + "/"):
                folder_db_rows.append(r)

        # 1. Detect and remove deleted files
        if remove_deleted and folder_db_rows:
            deleted_ids = []
            for r in folder_db_rows:
                r_norm = _normalize_path(r["file_path"])
                if r_norm not in disk_paths_set and not os.path.exists(r["file_path"]):
                    deleted_ids.append(r["id"])

            if deleted_ids:
                from backend.config import THUMB_DIR
                from backend.search import SemanticSearch

                delete_images(conn, deleted_ids)
                for iid in deleted_ids:
                    tpath = THUMB_DIR / f"{iid}.jpg"
                    if tpath.exists():
                        try:
                            tpath.unlink()
                        except Exception:
                            pass
                try:
                    SemanticSearch().delete(deleted_ids)
                except Exception as e:
                    print(f"Error removing deleted ids from vector index: {e}")

                stats["deleted_pruned"] = len(deleted_ids)
                del_set = set(deleted_ids)
                folder_db_rows = [r for r in folder_db_rows if r["id"] not in del_set]

        # 2. Existing lookups
        existing_hashes = {r["file_hash"] for r in all_db_rows}
        folder_hashes = {r["file_hash"] for r in folder_db_rows}
        folder_paths_map = {_normalize_path(r["file_path"]): r for r in folder_db_rows}

        total = len(image_paths)
        for idx, fpath in enumerate(image_paths, start=1):
            if callback:
                try:
                    callback(idx, total, Path(fpath).name)
                except Exception:
                    pass
            try:
                fhash = hash_file(fpath)
                fsize = os.path.getsize(fpath)
                fpath_norm = _normalize_path(fpath)

                if rescan_mode == "full":
                    # In full rescan: re-process existing images in this folder
                    matched_row = folder_paths_map.get(fpath_norm)
                    if not matched_row and fhash in folder_hashes:
                        for r in folder_db_rows:
                            if r["file_hash"] == fhash:
                                matched_row = r
                                break

                    if matched_row:
                        conn.execute(
                            """UPDATE images SET
                               has_metadata = 0,
                               has_description = 0,
                               is_embedded = 0,
                               raw_description = NULL,
                               enriched_text = NULL,
                               file_size = ?,
                               updated_at = datetime('now')
                               WHERE id = ?""",
                            (fsize, matched_row["id"])
                        )
                        stats["reprocessed"] += 1
                        continue

                # In incremental mode or new images:
                if fhash in existing_hashes:
                    stats["duplicate"] += 1
                    continue

                existing_hashes.add(fhash)
                upsert_image(conn, file_path=fpath, file_hash=fhash, file_size=fsize)
                stats["new"] += 1

            except Exception as e:
                stats["errors"] += 1

    return stats
