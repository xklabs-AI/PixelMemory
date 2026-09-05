"""
Ingest pipeline: scan → extract metadata (CPU) ∥ describe (GPU) → enrich → embed.

Usage:
    python -m backend.ingest /path/to/photos [--rescan] [--skip-describe]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import os
import threading
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from backend.config import DATA_DIR, CPU_WORKERS, BATCH_SIZE
from backend.db import (
    init_db, get_conn, get_pending_metadata, get_pending_descriptions,
    get_pending_embeds, update_metadata, update_description, mark_embedded,
    get_image_by_id, get_stats,
)
from backend.scanner import scan_directory
from backend.metadata import process_metadata
from backend.describer import ImageDescriber, enrich_description
from backend.search import SemanticSearch


class ImportProgressTracker:
    """Thread-safe tracker for the import and AI analysis pipeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self.status = "idle"  # idle | running | completed | error | cancelled
        self.phase = "idle"   # scanning | metadata | describing | embedding | done
        self.phase_label = "Ready"
        self.current_folder = ""
        self.queue_length = 0
        self.current_file = ""
        self.current_step = 0
        self.total_steps = 0
        self.percent = 0.0
        self.started_at = 0.0
        self.phase_started_at = 0.0
        self.eta_seconds = None
        self.speed_label = ""
        self.error_message = None
        self.stats = {
            "found": 0,
            "new": 0,
            "duplicate": 0,
            "errors": 0,
            "metadata_done": 0,
            "described": 0,
            "embedded": 0,
        }
        self.cancel_requested = False

    def set_queue_info(self, queue_length: int, folder_name: str):
        with self._lock:
            self.queue_length = queue_length
            self.current_folder = folder_name

    def start(self, folder: str):
        with self._lock:
            self.status = "running"
            self.phase = "scanning"
            self.phase_label = f"Scanning '{Path(folder).name}' for images..."
            self.current_folder = Path(folder).name
            self.current_file = Path(folder).name
            self.current_step = 0
            self.total_steps = 0
            self.percent = 0.0
            self.started_at = time.time()
            self.phase_started_at = time.time()
            self.eta_seconds = None
            self.speed_label = "Starting scan..."
            self.error_message = None
            self.cancel_requested = False
            self.stats = {k: 0 for k in self.stats}

    def update_scan(self, current: int, total: int, filename: str):
        with self._lock:
            self.current_step = current
            self.total_steps = total
            self.current_file = filename
            pct = (current / max(1, total)) * 5.0  # Scan is 0-5%
            self.percent = min(5.0, max(0.0, pct))
            self.phase_label = f"Scanning directory ({current}/{total} images found)"

    def update_metadata(self, current: int, total: int, filename: str):
        with self._lock:
            self.phase = "metadata"
            self.current_step = current
            self.total_steps = total
            self.current_file = filename
            self.stats["metadata_done"] = current
            # Metadata is 5% - 15%
            pct = 5.0 + (current / max(1, total)) * 10.0
            self.percent = min(15.0, pct)
            self.phase_label = f"Extracting GPS & EXIF metadata ({current}/{total})"

    def update_describing(self, current: int, total: int, filename: str, skip_describe: bool = False):
        with self._lock:
            self.phase = "describing"
            self.current_step = current
            self.total_steps = total
            self.current_file = filename
            self.stats["described"] = current

            # Describing is 15% - 90%
            pct = 15.0 + (current / max(1, total)) * 75.0
            self.percent = min(90.0, pct)

            # Dynamic ETA calculation
            elapsed_phase = time.time() - self.phase_started_at
            if current > 0:
                sec_per_img = elapsed_phase / current
                remaining_imgs = max(0, total - current)
                self.eta_seconds = remaining_imgs * sec_per_img
                self.speed_label = f"{sec_per_img:.1f}s / photo"
            model_tag = "LLaVA:7b" if "llava" in str(getattr(self, "vlm_model", "")).lower() else "Moondream2"
            self.phase_label = f"AI Vision captioning with {model_tag} ({current}/{total})"

    def update_embedding(self, current: int, total: int, filename: str):
        with self._lock:
            self.phase = "embedding"
            self.current_step = current
            self.total_steps = total
            self.current_file = filename
            self.stats["embedded"] = current
            # Embedding is 90% - 100%
            pct = 90.0 + (current / max(1, total)) * 10.0
            self.percent = min(99.0, pct)
            self.phase_label = f"Embedding semantic vectors into Zvec ({current}/{total})"
            self.eta_seconds = 1.0

    def complete(self, stats: dict):
        with self._lock:
            self.status = "completed"
            self.phase = "done"
            self.phase_label = "Analysis & Ingestion Complete!"
            self.current_file = ""
            self.percent = 100.0
            self.eta_seconds = 0.0
            self.speed_label = "Finished"
            self.stats.update(stats)

    def fail(self, error: str):
        with self._lock:
            self.status = "error"
            self.phase = "error"
            self.phase_label = "Import failed"
            self.error_message = str(error)

    def cancel(self):
        with self._lock:
            self.cancel_requested = True
            self.status = "cancelled"
            self.phase_label = "Import cancelled"

    def format_eta(self) -> str:
        if self.status == "completed":
            return "Complete"
        if self.eta_seconds is None or self.eta_seconds <= 0:
            return "Calculating..."
        eta = int(self.eta_seconds)
        if eta < 60:
            return f"~{eta}s remaining"
        mins, secs = divmod(eta, 60)
        return f"~{mins}m {secs}s remaining"

    def to_dict(self) -> dict:
        with self._lock:
            elapsed = time.time() - self.started_at if self.started_at > 0 else 0
            return {
                "status": self.status,
                "phase": self.phase,
                "phase_label": self.phase_label,
                "current_folder": self.current_folder,
                "queue_length": self.queue_length,
                "current_file": self.current_file,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "percent": round(self.percent, 1),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds is not None else None,
                "eta_human": self.format_eta(),
                "speed_label": self.speed_label,
                "stats": dict(self.stats),
                "error_message": self.error_message,
            }


# Global singleton tracker
tracker = ImportProgressTracker()


def run_metadata_extraction(callback: Optional[Callable] = None) -> int:
    """Extract EXIF + geocode for all pending images. CPU-bound, threaded."""
    with get_conn() as conn:
        pending = get_pending_metadata(conn, limit=10_000)

    if not pending:
        return 0

    processed = 0
    total = len(pending)

    def _do_one(row):
        meta = process_metadata(row["id"], row["file_path"])
        return row["id"], row["file_path"], meta

    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as pool:
        futures = {pool.submit(_do_one, row): row for row in pending}
        for future in as_completed(futures):
            try:
                image_id, file_path, meta = future.result()
                with get_conn() as conn:
                    update_metadata(conn, image_id, **meta)
                processed += 1
                if callback:
                    callback(processed, total, Path(file_path).name)
            except Exception as e:
                row = futures[future]
                print(f"  [meta error] {row['file_path']}: {e}")

    return processed


def run_description_generation(vlm_model: Optional[str] = None, callback: Optional[Callable] = None) -> int:
    """Generate VLM descriptions for all pending images. GPU-bound."""
    with get_conn() as conn:
        pending = get_pending_descriptions(conn, limit=10_000)

    if not pending:
        return 0

    describer = ImageDescriber(model_name=vlm_model)
    describer.load()

    items = [(row["id"], row["file_path"]) for row in pending]
    total = len(items)
    described = 0

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]
        results = describer.describe_batch(batch)

        with get_conn() as conn:
            for image_id, raw_desc in results:
                row = get_image_by_id(conn, image_id)
                meta = {
                    "date_taken": row["date_taken"],
                    "place_name": row["place_name"],
                    "camera_model": row["camera_model"],
                }
                enriched = enrich_description(raw_desc, meta)
                update_description(conn, image_id, raw=raw_desc, enriched=enriched)
                described += 1
                if callback:
                    callback(described, total, Path(row["file_path"]).name)

    return described


def run_embedding(callback: Optional[Callable] = None) -> int:
    """Embed all described-but-not-yet-embedded images into Zvec."""
    with get_conn() as conn:
        pending = get_pending_embeds(conn, limit=10_000)

    if not pending:
        return 0

    search = SemanticSearch()
    items = [(row["id"], row["enriched_text"]) for row in pending]
    total = len(items)
    embedded = 0

    for batch_start in range(0, len(items), 256):
        batch = items[batch_start : batch_start + 256]
        search.add_batch(batch)

        with get_conn() as conn:
            for image_id, _ in batch:
                mark_embedded(conn, image_id)
                embedded += 1
                if callback:
                    callback(embedded, total, f"Image #{image_id}")

    return len(items)


def execute_import(directory: str, skip_describe: bool = False, vlm_model: Optional[str] = None):
    """Executes the full import pipeline in sequence, updating tracker."""
    try:
        tracker.start(directory)
        tracker.vlm_model = vlm_model or "moondream:1.8b"

        # 1. Scan directory
        scan_stats = scan_directory(directory, callback=tracker.update_scan)
        tracker.stats.update(scan_stats)

        # 2. Metadata extraction & AI description only for newly discovered files
        if scan_stats.get("new", 0) > 0:
            tracker.phase_started_at = time.time()
            tracker.phase = "metadata"
            meta_count = run_metadata_extraction(callback=tracker.update_metadata)
            tracker.stats["metadata_done"] = meta_count

            # 3. AI Vision Descriptions (applied only to new photos)
            if not skip_describe:
                tracker.phase_started_at = time.time()
                tracker.phase = "describing"
                desc_count = run_description_generation(vlm_model=vlm_model, callback=tracker.update_describing)
                tracker.stats["described"] = desc_count

                # 4. Vector Embedding
                tracker.phase_started_at = time.time()
                tracker.phase = "embedding"
                embed_count = run_embedding(callback=tracker.update_embedding)
                tracker.stats["embedded"] = embed_count
        else:
            tracker.phase_label = f"Rescan complete: all {scan_stats.get('duplicate', 0)} photos are already up-to-date."

        with get_conn() as conn:
            final_stats = get_stats(conn)
        tracker.complete(final_stats)


    except Exception as e:
        print(f"Import failed with error: {e}")
        tracker.fail(str(e))


class ImportQueueManager:
    """Manages sequential background imports with a FIFO queue."""

    def __init__(self, tracker: ImportProgressTracker):
        from collections import deque
        self.tracker = tracker
        self.queue = deque()
        self.lock = threading.Lock()
        self.worker_thread = None
        self.is_running = False

    def enqueue(self, directory: str, skip_describe: bool = False, vlm_model: Optional[str] = None) -> dict:
        with self.lock:
            # Avoid duplicate queuing of exact same folder
            for d, _, _ in self.queue:
                if Path(d).resolve() == Path(directory).resolve():
                    pos = [x[0] for x in self.queue].index(d) + 1
                    return {
                        "status": "already_queued",
                        "queue_position": pos,
                        "queue_length": len(self.queue),
                    }

            self.queue.append((directory, skip_describe, vlm_model))
            queue_len = len(self.queue)

            if not self.is_running:
                self.is_running = True
                self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self.worker_thread.start()
                return {
                    "status": "started",
                    "queue_position": 1,
                    "queue_length": 1,
                }
            else:
                self.tracker.set_queue_info(queue_len - 1, self.tracker.current_folder)
                return {
                    "status": "queued",
                    "queue_position": queue_len,
                    "queue_length": queue_len,
                }

    def _worker_loop(self):
        while True:
            with self.lock:
                if not self.queue:
                    self.is_running = False
                    return
                directory, skip_describe, vlm_model = self.queue.popleft()
                remaining = len(self.queue)

            self.tracker.set_queue_info(remaining, Path(directory).name)
            execute_import(directory, skip_describe, vlm_model=vlm_model)

            if self.tracker.cancel_requested:
                with self.lock:
                    self.queue.clear()
                    self.is_running = False
                return

    def get_queue_info(self) -> dict:
        with self.lock:
            return {
                "is_running": self.is_running,
                "queue_length": len(self.queue),
                "queued_folders": [Path(d).name for d, _, _ in self.queue]
            }

    def cancel(self):
        with self.lock:
            self.queue.clear()
        self.tracker.cancel()


# Global queue manager instance
queue_manager = ImportQueueManager(tracker)


def start_background_import(directory: str, skip_describe: bool = False, vlm_model: Optional[str] = None) -> dict:
    """Enqueue directory to import pipeline."""
    return queue_manager.enqueue(directory, skip_describe, vlm_model=vlm_model)


def main():
    parser = argparse.ArgumentParser(description="PixelMemory ingest pipeline")
    parser.add_argument("directory", help="Root directory to scan for images")
    parser.add_argument("--rescan", action="store_true", help="Re-scan even if images exist in DB")
    parser.add_argument("--skip-describe", action="store_true", help="Skip VLM description (metadata only)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()

    t0 = time.time()

    # Phase 1: Scan
    print(f"\n{'='*60}")
    print(f"Phase 1: Scanning {args.directory}")
    print(f"{'='*60}")
    scan_stats = scan_directory(args.directory)
    print(f"  Found: {scan_stats['found']}  New: {scan_stats['new']}  "
          f"Duplicate: {scan_stats['duplicate']}  Errors: {scan_stats['errors']}")

    # Phase 2: Metadata extraction (CPU) — runs first so descriptions can use it
    print(f"\n{'='*60}")
    print("Phase 2: Extracting metadata (CPU)")
    print(f"{'='*60}")
    meta_count = run_metadata_extraction()
    print(f"  Processed metadata for {meta_count} images")

    # Phase 3: VLM descriptions (GPU)
    if not args.skip_describe:
        print(f"\n{'='*60}")
        print("Phase 3: Generating descriptions (GPU)")
        print(f"{'='*60}")
        desc_count = run_description_generation()
        print(f"  Described {desc_count} images")

        # Phase 4: Embed into vector DB
        print(f"\n{'='*60}")
        print("Phase 4: Embedding descriptions")
        print(f"{'='*60}")
        embed_count = run_embedding()
        print(f"  Embedded {embed_count} descriptions")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    with get_conn() as conn:
        stats = get_stats(conn)
    print(f"  Total images: {stats['total']}")
    print(f"  Metadata extracted: {stats['metadata_done']}")
    print(f"  Described: {stats['described']}")
    print(f"  Embedded: {stats['embedded']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
