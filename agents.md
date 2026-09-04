# PixelMemory — Agent Architecture

## Overview

PixelMemory is decomposed into specialized agents, each owning a distinct responsibility in the pipeline. Agents communicate through the shared SQLite database (state machine) and the filesystem (thumbnails, originals). Each agent is independently restartable — pipeline state lives in the DB, not in memory.

```
┌──────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                              │
│  Reads CLI args / config → launches agents in order              │
│  Monitors progress → reports stats → handles graceful shutdown   │
└──────┬──────────┬──────────────┬─────────────┬───────────────────┘
       │          │              │             │
       ▼          ▼              ▼             ▼
  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Scanner │ │ Metadata │ │Describer │ │ Embedder │
  │  Agent  │ │  Agent   │ │  Agent   │ │  Agent   │
  │  (CPU)  │ │  (CPU)   │ │  (GPU)   │ │ (CPU/GPU)│
  └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
       │           │            │             │
       ▼           ▼            ▼             ▼
  ┌────────────────────────────────────────────────┐
  │              SQLite (shared state)              │
  │  images table: per-row pipeline status flags    │
  │  metadata_done | description_done | embedded    │
  └────────────────────────────────────────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │   ChromaDB    │
                 │ (vector store)│
                 └───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Search Agent     │
              │  (FastAPI server)   │
              │  serves query API   │
              │  + frontend UI      │
              └─────────────────────┘
```

---

## Agent Definitions

### 1. Orchestrator

**Role:** Pipeline coordinator. Launches agents in dependency order, monitors progress, handles interrupts.

**Owns:**
- CLI argument parsing (directory path, flags)
- Agent sequencing: Scanner → Metadata Agent → Describer Agent → Embedder Agent
- Progress reporting (totals, rates, ETA)
- Graceful shutdown on SIGINT/SIGTERM (finish current batch, commit, exit)

**State contract:**
- Reads `images` table aggregate counts to determine what work remains
- Does not modify rows directly — delegates to child agents

**Failure handling:**
- If any agent crashes, the orchestrator logs the error and reports progress so far
- Pipeline is always resumable: re-running skips completed work via status flags

---

### 2. Scanner Agent

**Role:** Filesystem discovery and deduplication.

**Owns:**
- Recursive directory walk filtered by `SUPPORTED_EXTENSIONS`
- Content hashing (xxHash) for deduplication
- Initial row insertion into `images` table

**Input:** A root directory path (or list of paths).

**Output:** Rows in `images` table with `file_path`, `file_hash`, `file_size`. All status flags at 0.

**Concurrency model:** Single-threaded. Disk I/O bound — parallelism doesn't help and risks thrashing on spinning drives.

**Decisions made by this agent:**
- Skip files whose content hash already exists in DB (exact duplicate, different path)
- Skip files that can't be read (permission errors, corrupt headers) — log and continue
- Update path if hash matches an existing row but path differs (file was moved)

**Does NOT do:**
- Open images as pixel data
- Parse EXIF
- Make any judgment about image content

---

### 3. Metadata Agent

**Role:** Extract structured metadata from image files and resolve locations.

**Owns:**
- EXIF parsing (date, GPS, camera, orientation) via `exifread`
- GPS coordinate conversion (DMS → decimal, hemisphere sign)
- Offline reverse geocoding (GPS → city, region, country) via `reverse_geocoder`
- Thumbnail generation (Pillow resize → JPEG cache)

**Input:** Rows where `metadata_done = 0`.

**Output:** Updated rows with EXIF fields populated, `metadata_done = 1`. Thumbnail files on disk.

**Concurrency model:** Thread pool (4 workers default). Each image is independent — pure CPU work, no shared mutable state except DB writes (serialized by SQLite WAL).

**Decisions made by this agent:**
- Fallback chain for date fields (DateTimeOriginal preferred)
- Hemisphere sign application for GPS
- Mode conversion for thumbnails (RGBA → RGB)
- If no EXIF exists, mark `metadata_done = 1` anyway with null fields — don't block the pipeline

**Does NOT do:**
- Visual analysis of image content
- Any GPU work

---

### 4. Describer Agent

**Role:** Generate natural language descriptions of images using a vision-language model.

**Owns:**
- VLM model loading and GPU memory management (Moondream2)
- Image preprocessing (open, convert to RGB, handle orientation)
- Prompt construction and inference
- Description enrichment: prepend date/location/camera context to raw VLM output

**Input:** Rows where `description_done = 0`. Reads metadata fields (date_taken, place_name, camera_model) from the same row to build enriched text.

**Output:** Updated rows with `raw_description`, `enriched_text`, `description_done = 1`.

**Concurrency model:** Single process, single GPU. Sequential image processing. Batched commits to DB (every `BATCH_SIZE` images).

**Resource profile:**
- GPU: ~3-4 GB VRAM for Moondream2 (fits comfortably on either the 4070 or 5060 Ti 16GB)
- Throughput: ~2-5 images/second depending on resolution and model

**Decisions made by this agent:**
- Prompt wording (configured in `config.py`, iterable)
- Enrichment format: `"{date}, {place}, taken with {camera} — {description}"`
- On inference failure: store `"[description failed: {error}]"` rather than crashing — lets pipeline continue, failures are greppable

**Dependency:** Should run AFTER Metadata Agent completes (or at least after the target image's metadata is done), so enrichment has date/location context. The orchestrator enforces this ordering.

**Does NOT do:**
- Embedding or vector storage
- Search

---

### 5. Embedder Agent

**Role:** Convert enriched text descriptions into vector embeddings and store in ChromaDB.

**Owns:**
- Embedding model loading (`all-MiniLM-L6-v2` via sentence-transformers)
- Batch encoding of text → vectors
- ChromaDB upsert (persistent collection, cosine similarity, HNSW index)

**Input:** Rows where `description_done = 1 AND embedded = 0`. Reads `enriched_text`.

**Output:** Vectors stored in ChromaDB, `embedded = 1` in SQLite.

**Concurrency model:** Single process. Batch size 256 for encoding efficiency. Can run on CPU or GPU — the embedding model is small enough that CPU is fast.

**Resource profile:**
- RAM: ~500 MB for the embedding model
- Throughput: ~500-1000 embeddings/second on CPU

**Decisions made by this agent:**
- Batch size for encoding (256 — balances memory and throughput)
- ChromaDB collection configuration (cosine space, HNSW)

**Does NOT do:**
- Search or query handling
- Any image processing

---

### 6. Search Agent

**Role:** Handle user queries at runtime. Serves the API and frontend.

**Owns:**
- Query embedding (same model as Embedder Agent for consistency)
- ChromaDB approximate nearest neighbor search
- Result enrichment: join vector hits with SQLite metadata for the response
- Thumbnail and original image serving
- Static frontend (HTML/CSS/JS)

**Input:** User's natural language query string via HTTP.

**Output:** Ranked list of `{image_id, score, enriched_text, date_taken, place_name, ...}`.

**API surface:**
| Endpoint | Purpose |
|---|---|
| `GET /api/search?q=...&top_k=40` | Semantic search, returns ranked results |
| `GET /api/thumb/{id}` | Cached thumbnail JPEG |
| `GET /api/original/{id}` | Original file from disk |
| `GET /api/image/{id}` | Full metadata for one image |
| `GET /api/stats` | Pipeline progress counts |
| `GET /` | Frontend HTML |

**Concurrency model:** Async FastAPI with uvicorn. Multiple concurrent queries supported. ChromaDB and SQLite reads are thread-safe.

**Does NOT do:**
- Ingest, describe, or embed — read-only at runtime
- Modify the database

---

## State Machine

Each image row progresses through a linear state machine. Agents pick up work based on flag values.

```
┌───────────┐     ┌───────────────┐     ┌──────────────────┐     ┌──────────┐
│  Scanned  │ ──► │ Metadata Done │ ──► │ Description Done │ ──► │ Embedded │
│  (0,0,0)  │     │   (1,0,0)     │     │    (1,1,0)       │     │ (1,1,1)  │
└───────────┘     └───────────────┘     └──────────────────┘     └──────────┘
    ▲                                                                  │
    │                Scanner writes                   Search Agent reads│
    └──────────────────────────────────────────────────────────────────┘
```

Flags: `(metadata_done, description_done, embedded)`

This design means:
- Any agent can crash and restart — it queries for rows matching its input state
- No in-memory queues or message passing — the DB is the queue
- Progress is always visible via `SELECT COUNT(*) ... GROUP BY flags`
- Multiple ingest runs are additive — new images enter at (0,0,0), existing ones are skipped

---

## GPU Assignment Strategy

With two GPUs available (RTX 4070 12GB + RTX 5060 Ti 16GB):

| Agent | Device | Rationale |
|---|---|---|
| Scanner | CPU | Disk I/O only |
| Metadata | CPU | EXIF parsing, reverse geocoding, thumbnail resize |
| Describer | GPU 0 or 1 | VLM inference — assign via `CUDA_VISIBLE_DEVICES` |
| Embedder | CPU (or GPU 1) | Embedding model is small, CPU is fast enough |
| Search | CPU (+ GPU for query embedding) | Query embedding is a single vector, negligible GPU time |

For maximum throughput during initial ingest of a large archive, the Describer Agent is the bottleneck. Options:

- **Single GPU:** Run Describer on the 5060 Ti (more VRAM headroom for larger VLMs if desired).
- **Dual GPU:** Run two Describer Agent instances, one per GPU, partitioning work by image ID parity or range. Requires care with DB writes but SQLite WAL handles concurrent writers.

---

## Error Handling Philosophy

- **Never crash the pipeline over a single image.** Log the error, mark the image with a greppable failure string, continue.
- **Always be resumable.** State lives in the DB, not in memory. Kill the process at any point, re-run, and it picks up where it left off.
- **Fail loud, recover quiet.** Errors print to stderr with the file path. Successful processing is silent except for the progress bar.
- **Don't block downstream on upstream failures.** If metadata extraction fails for an image, the Describer Agent still processes it — the enriched text just won't have date/location context. A missing description is better than no description.

---

## Future Agents (Not Yet Implemented)

### Face Clustering Agent
- Run face detection (RetinaFace or SCRFD) on each image
- Extract face embeddings (ArcFace)
- Cluster by embedding similarity
- Let users name clusters → search by person name
- Would add a `faces` table and a `face_clusters` table to the DB

### Query Parser Agent
- Decompose natural language queries into structured filters + semantic residual
- `"beach photos from 2019 near LA"` → `{semantic: "beach", date: ["2019-01-01", "2019-12-31"], geo: {lat: 34.05, lon: -118.24, radius_km: 50}}`
- Could use a small local LLM or rule-based extraction
- Feeds into hybrid search: vector for semantic, SQL for structured

### Watch Agent
- Background daemon monitoring directories for new files
- On new file detection: trigger Scanner → Metadata → Describer → Embedder
- Would use filesystem events (inotify on Linux, FSEvents on macOS)
- Keeps the database live without manual re-runs

### Export Agent
- Dump text-image pairs in training-ready formats
- Output: folder of images + matching `.txt` caption files (for LoRA/Dreambooth trainers)
- Filtered export: only images matching a query, a date range, or a face cluster
- Feeds into the family photo synthesis pipeline
