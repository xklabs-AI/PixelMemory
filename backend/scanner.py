"""Recursively scan directories for images, hash and deduplicate."""

import os
from pathlib import Path

import xxhash
from tqdm import tqdm

from backend.config import SUPPORTED_EXTENSIONS, HASH_CHUNK_SIZE
from backend.db import get_conn, upsert_image


def hash_file(path: str) -> str:
    """Fast content hash using xxHash (non-cryptographic, very fast)."""
    h = xxhash.xxh64()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def scan_directory(root: str | Path, callback=None) -> dict:
    """
    Walk a directory tree, find images, hash them, insert into DB.
    Returns stats: {found, new, duplicate, errors}.
    Optional callback(current, total, file_name) for live progress tracking.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    # Collect all image paths first for the progress bar
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(dirpath, fname))

    stats = {"found": len(image_paths), "new": 0, "duplicate": 0, "errors": 0}

    # Track hashes we've already seen in this run to skip dupes early
    seen_hashes: set[str] = set()

    with get_conn() as conn:
        # Pre-load existing hashes from DB
        rows = conn.execute("SELECT file_hash FROM images").fetchall()
        for r in rows:
            seen_hashes.add(r["file_hash"])

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

                if fhash in seen_hashes:
                    stats["duplicate"] += 1
                    continue

                seen_hashes.add(fhash)
                upsert_image(conn, file_path=fpath, file_hash=fhash, file_size=fsize)
                stats["new"] += 1

            except Exception as e:
                stats["errors"] += 1

    return stats
