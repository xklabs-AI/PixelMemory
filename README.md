# PixelMemory — Local Semantic Photo Search

> **Find any photo using natural language.** Fully offline and privacy-first — powered by local Vision AI, offline reverse geocoding, and vector search. No photos or metadata ever leave your machine.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Ollama Moondream](https://img.shields.io/badge/VLM-Moondream2%20(Ollama)-purple.svg)](https://ollama.com/)

---

## ✨ Features

- 🔒 **100% Private & Local** — Zero cloud dependencies, zero data leaks. All AI inference and vector indexing run locally on your hardware.
- 🧠 **Deep Visual Understanding** — Powered by **Moondream2** (via Ollama or local PyTorch) to describe subjects, actions, colors, scenery, objects, and mood.
- 📍 **Offline GPS & Reverse Geocoding** — Automatically extracts EXIF GPS coordinates and maps them to human-readable place names (city, region, country) completely offline.
- 🎯 **High-Precision Neural Search** — Aspect conjunction and dynamic elbow cutoff algorithms eliminate unrelated false positives while retaining high recall.
- ⚡ **Non-Blocking Background Ingest** — Google Drive style floating progress dock with live speed, step counter, dynamic ETA countdown, and sequential folder queueing.
- 📁 **Albums & Curation** — Create custom albums, assign photos with one click, and browse curated collections.
- 🏷️ **Dynamic Tag Explorer** — Automatic camera model, location, and year tag extraction for quick filtering.
- 🚀 **Unified Launcher** — Single-command setup and diagnostics with `python launch.py`.

---

## 🏛️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INGEST & ANALYSIS PIPELINE                      │
│                                                                        │
│  Directory Scanner ──► EXIF / GPS Extractor (CPU) ──────────┐          │
│                              │                              ▼          │
│                        Offline Reverse Geocode         SQLite DB       │
│                                                             ▲          │
│  Directory Scanner ──► Moondream2 Vision VLM (Local GPU) ───┘          │
│                        (detailed visual captioning)                    │
│                                                                        │
│  Metadata Enrichment ──► sentence-transformers ──► ChromaDB Vector DB │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                          SEARCH & DISCOVERY                            │
│                                                                        │
│  Natural Language Query ──► Neural Conjunction Filter ──► ChromaDB     │
│                                      │                                 │
│                               Ranked Results                           │
│                                      │                                 │
│                             FastAPI REST API                           │
│                                      │                                 │
│                         Web UI (Glassmorphic SPA)                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: (Recommended) Installed and running locally
  ```bash
  ollama pull moondream
  ```
  *(Alternative: Moondream2 can also run directly via PyTorch/HuggingFace if configured in `backend/config.py`)*
- **GPU**: NVIDIA GPU with 4GB+ VRAM recommended for fast AI vision captioning (CPU fallback supported).

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/xklabs-AI/PixelMemory.git
cd PixelMemory
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launch the Application
Run the unified interactive launcher:
```bash
python launch.py
```
Or start the server directly:
```bash
python -m backend.server
```

Open your browser at:
👉 **`http://localhost:8642`**

---

## 📥 Ingesting Your Photos

You can import photos directly from the **Web Interface**:
1. Click **Import Photos** in the sidebar.
2. Enter the path to any local directory of photos (e.g. `C:\Users\Username\Pictures` or `/home/user/Photos`).
3. Click **Start Import**.
4. The **Floating Progress Dock** will track progress in the background while you continue searching and browsing. Additional folder imports will be automatically queued sequentially!

You can also run ingest via CLI:
```bash
python -m backend.ingest /path/to/your/photos
```

---

## ⚙️ Configuration

Key settings can be modified in [`backend/config.py`](backend/config.py):

| Setting | Default | Description |
|---|---|---|
| `DATA_DIR` | `~/.pixelmemory` | Root data directory for SQLite DB, ChromaDB, and thumbnails |
| `DB_PATH` | `~/.pixelmemory/pixelmemory.db` | SQLite library database path |
| `CHROMA_DIR` | `~/.pixelmemory/chroma` | ChromaDB vector persistence directory |
| `USE_OLLAMA` | `True` | Whether to use Ollama for Moondream2 inference |
| `OLLAMA_MODEL` | `moondream:1.8b` | Ollama model tag |
| `EMBEDDING_MODEL`| `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
| `PORT` | `8642` | Web server port |

---

## 🛡️ Privacy & Security

- **Strictly Offline**: All processing occurs locally on your machine.
- **No Cloud Telemetry**: PixelMemory makes zero external API requests during search or ingestion.
- **Gitignored User Data**: Databases, photo directories, and vector embeddings are gitignored by default.

---

## 📄 License

This project is open-source software licensed under the [MIT License](LICENSE).
