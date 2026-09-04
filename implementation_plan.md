# PixelMemory — Implementation Plan

## Vision

A fully local, privacy-preserving semantic photo search tool. Users point it at years of phone backup images on a hard drive and get natural language search: "show me people near the Grand Canyon from 2016" returns ranked results by fusing visual understanding, GPS metadata, and temporal context — all processed on local GPU, nothing leaves the machine.

---

## Phase 1: Foundation (Week 1)

**Goal:** Scan a photo directory, extract metadata, and store everything in a queryable database.

### 1.1 Project scaffolding
- Python project structure with `backend/` and `frontend/` directories
- Virtual environment, `requirements.txt`, config module
- SQLite database with WAL mode for concurrent read/write

### 1.2 Recursive scanner
- Walk directory tree, filter by supported extensions (JPEG, PNG, HEIC, WEBP, TIFF, BMP)
- Content-hash each file with xxHash for fast deduplication
- Insert unique images into `images` table with path, hash, size
- Progress bar via tqdm, error tolerance (skip unreadable files, keep going)

### 1.3 EXIF metadata extraction
- Extract date taken (DateTimeOriginal → DateTimeDigitized → DateTime fallback chain)
- Extract GPS coordinates, convert DMS to decimal, apply hemisphere sign
- Extract camera make/model and orientation
- Threaded execution (4 CPU workers) since this is I/O + CPU bound

### 1.4 Offline reverse geocoding
- Use `reverse_geocoder` library (GeoNames k-d tree, no network calls)
- Map GPS coordinates → city, region, country, place_name
- Run in same CPU thread pool as EXIF extraction

### 1.5 Thumbnail generation
- Pillow resize to 320×320, JPEG output at quality 80
- Handle mode conversion (RGBA/palette → RGB)
- Cache in `~/.pixelmemory/thumbnails/{image_id}.jpg`

**Deliverable:** Run `python -m backend.ingest /path/to/photos` and get a SQLite database with all images indexed, metadata extracted, thumbnails generated. No GPU required yet.

---

## Phase 2: Visual Understanding (Week 2)

**Goal:** Generate rich text descriptions of every image using a local vision-language model.

### 2.1 Moondream2 integration
- Load model on GPU (RTX 4070 or 5060 Ti)
- Single-image inference: encode image → query with detailed prompt
- Prompt engineering for rich descriptions: scene, people, objects, colors, lighting, text, mood

### 2.2 Enriched description construction
- Prepend metadata context to VLM output
- Format: `"{month} {year}, {place_name}, taken with {camera} — {visual description}"`
- This is the core insight: all search signals collapse into a single text string that gets embedded together

### 2.3 Pipeline orchestration
- CPU metadata extraction and GPU description generation share the pipeline but run sequentially (metadata first, so enrichment has context)
- Batch processing with configurable batch size (16 default, adjust for VRAM)
- Checkpoint progress per image in DB — pipeline is resumable on crash
- Status flags: `metadata_done`, `description_done`, `embedded`

### 2.4 VLM prompt iteration
- Run on a sample of ~100 diverse images
- Evaluate description quality: are faces counted? Is indoor/outdoor detected? Are colors and lighting captured?
- Iterate prompt wording until descriptions are specific enough for semantic differentiation
- Consider: if two visually different images produce nearly identical descriptions, the prompt needs more specificity

**Deliverable:** Every image in the database has a `raw_description` (VLM output) and `enriched_text` (metadata + VLM). Pipeline resumes cleanly if interrupted.

---

## Phase 3: Semantic Search (Week 3)

**Goal:** Embed descriptions into a vector database and serve search results through a web API.

### 3.1 Embedding pipeline
- Embed `enriched_text` using `all-MiniLM-L6-v2` via sentence-transformers
- Store vectors in ChromaDB (persistent, cosine similarity, HNSW index)
- Batch embedding (256 at a time) with progress tracking
- Mark `embedded = 1` in SQLite after successful storage

### 3.2 Search API (FastAPI)
- `GET /api/search?q=...&top_k=40` — embed query, ANN search, return ranked results with metadata
- `GET /api/thumb/{id}` — serve cached thumbnails
- `GET /api/original/{id}` — serve original files from disk
- `GET /api/stats` — pipeline progress (total, described, embedded)
- `GET /api/image/{id}` — full metadata for detail view

### 3.3 Search quality tuning
- Test with queries that combine semantic + spatial + temporal signals
- Measure: does "beach sunset 2019" return beach photos from 2019, not beach photos from 2023?
- If temporal/spatial precision is weak, consider hybrid approach: vector search for candidates → SQL filter for date range and geo radius
- Tune `top_k` and similarity threshold for result quality vs. recall

**Deliverable:** `python -m backend.server` starts a FastAPI server. Hitting `/api/search?q=birthday+party` returns ranked image results with thumbnails.

---

## Phase 4: Frontend (Week 3–4)

**Goal:** A browser-based search interface that feels fast and purposeful.

### 4.1 Search experience
- Single search bar with example query pills for discoverability
- Results as a thumbnail grid with date, location, and similarity score
- Lazy-loaded images for performance on large result sets
- Keyboard shortcut: Enter to search

### 4.2 Detail view
- Click a result → modal with full-size image
- Show enriched description, date, location, GPS, camera, file path
- Original file served on demand (not preloaded)

### 4.3 Stats bar
- Show pipeline progress: total images, described, searchable
- Helps users understand whether ingest is complete

### 4.4 Polish
- Dark theme (photo-focused UI, reduce eye strain)
- Responsive grid (works on ultrawide and laptop screens)
- Error states: server not running, no results, ingest incomplete

**Deliverable:** Open `http://localhost:8642`, type a query, browse results, click for detail. Single HTML file, no build step.

---

## Phase 5: Hardening & Quality of Life (Week 4–5)

### 5.1 Incremental ingest
- Re-run scanner on same directory → only process new/changed files
- Detect moved files by hash (same content, new path → update path, skip re-describe)
- Optional `--rescan` flag for full re-index

### 5.2 HEIC handling
- Install `pillow-heif` for Apple HEIC/HEIF support
- Register with Pillow so `.heic` files open transparently
- Test with iPhone photo exports (the most common source of HEIC)

### 5.3 Duplicate management
- Current: skip exact content duplicates by hash
- Add: near-duplicate detection (same photo, slightly different crop or compression)
- Approach: perceptual hash (pHash) as a secondary dedup signal

### 5.4 Performance benchmarks
- Measure throughput on a real archive (target: 1,000+ images/hour on 4070)
- Profile bottlenecks: is it VLM inference, disk I/O, embedding, or DB writes?
- Optimize the slowest stage

### 5.5 Multi-directory support
- Allow multiple source directories in a single database
- Track which root each image belongs to
- Handle external drives that may not always be mounted

**Deliverable:** A robust tool that handles real-world photo archives with messy formats, duplicates, and incremental updates.

---

## Phase 6: Advanced Search (Week 5–6)

### 6.1 Hybrid search
- Parse structured filters from natural language queries
- Use a small local LLM or rule-based parser to extract: `{semantic: "...", date_range: [...], location_radius: {...}}`
- Vector search for semantic component → SQL filter for date/geo
- Intersect result sets for precision

### 6.2 Faceted browsing
- Add sidebar filters: year, month, location, camera
- Aggregate counts from SQLite for each facet
- Filter without re-running vector search (client-side filtering of existing results)

### 6.3 Map view
- Plot GPS-tagged results on an interactive map
- Cluster nearby results
- Click cluster → see photos from that area
- Use Leaflet.js with OpenStreetMap tiles (no API key needed)

### 6.4 Timeline view
- Horizontal timeline of results grouped by date
- Scroll through years/months
- Visual density indicator (months with more matches appear heavier)

**Deliverable:** Users can search, filter by date/location, browse on a map, and scan a timeline — all against a fully local database.

---

## Future Directions (Not Scoped)

- **Face clustering:** Run a face detection model, cluster by embedding, let users name clusters → search by person name
- **Album generation:** Auto-group related images into albums by event/location/time proximity
- **Dataset export:** Dump text-image pairs for diffusion model fine-tuning (LoRA training data)
- **Family photo synthesis:** Compose per-person LoRAs to generate group photos that never existed
- **Watch mode:** Background daemon that monitors directories for new photos and auto-ingests
- **Mobile companion:** PWA or lightweight mobile UI for searching from phone over LAN
