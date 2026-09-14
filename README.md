# PixelMemory — Local Semantic Photo Search

> **Your closest moments deserve better than a forgotten folder. Rediscover and relive your entire photo library with natural language search, face recognition & story timelines — powered by local Vision AI. No cloud, no telemetry, no compromise.

[![Python 3.10-3.12](https://img.shields.io/badge/Python-3.10%20--%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tauri v2](https://img.shields.io/badge/Desktop-Tauri%20v2%20(Rust)-orange.svg)](https://tauri.app/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Ollama Moondream2](https://img.shields.io/badge/Vision-Moondream2%20(1.8B)-purple.svg)](https://ollama.com/library/moondream)
[![Ollama Gemma 4](https://img.shields.io/badge/Stories-Gemma%204%20(2B)-red.svg)](https://ollama.com/library/gemma4)

---

## ✨ Core Features & Capabilities

- 🔒 **100% Private & Strictly Offline** — Zero cloud dependencies, zero external telemetry. All neural inference, facial detection, vector embeddings, and reverse geocoding run entirely on your local machine.
- 🖥️ **Native Desktop Application (Tauri v2)** — High-performance native desktop shell (`PixelMemory.exe` / macOS App) featuring a frameless glass header, custom window controls, and native File Explorer/Finder directory pickers.
- 🧠 **Deep Visual Understanding (Moondream2)** — Pre-configured to use **Moondream2** (`moondream`) via Ollama for ultra-fast GPU visual description (~0.8s/photo) capturing scene categories, objects, actions, clothing, colors, and mood.
- 🗣️ **Smart Natural Language Query Understanding** — Intelligent query intent parser automatically decomposes queries like *"photos in Tokyo last summer with Alice"* into semantic visual vectors, location filters, date/time ranges, and recognized people.
- 🗺️ **Interactive Geographic Map View** — Visualizes your photos on an interactive dark-mode world map with geographic clustering based on EXIF GPS metadata.
- 📖 **Personal Story Timelines (Gemma 4 2B)** — Pre-configured to use **Gemma 4** (`gemma4:e2b`) to transform chronological photo groups and memories into warm, personal first-person journal narratives.
- 👤 **On-Device Face Recognition & Clustering** — Built-in YuNet face detection and SFace deep facial embeddings for tagging friends, family, and pets without cloud biometric databases.
- 📍 **Offline Reverse Geocoding** — Automatically extracts EXIF GPS coordinates and maps them to human-readable place names (city, region, country) with zero network calls.
- 🔍 **Visual Similarity / "More Like This"** — Instant vector nearest-neighbor search to find visually and contextually similar memories from any photo in your archive.
- 📷 **Deep EXIF & Camera Metadata Inspector** — Full metadata breakdown showing camera model, lens, focal length, aperture, shutter speed, ISO, timestamp, and AI caption tags.
- ⚡ **Incremental Ingest & Folder Sync** — High-speed sequential photo indexing with file hash change detection, deleted photo pruning, and live ETA progress dock.

---

## 🤖 Pre-Configured Default AI Models & Tech Stack

PixelMemory is out-of-the-box optimized for consumer GPUs, Apple Silicon, and modern CPUs:

| Capability | Model / Engine | Speed / Resource | Purpose |
| :--- | :--- | :--- | :--- |
| **Vision (VLM)** | **`moondream`** (Moondream2 1.8B) | ~0.8s/photo · 1.8 GB VRAM | Comprehensive visual captioning for semantic search |
| **Stories (LLM)** | **`gemma4:e2b`** (Gemma 4 2B) | ~1-2s · ~2 GB VRAM | Warm first-person daily travel journals and narratives |
| **Query Intent** | **Fast NLP & Intent Parser** | Instant (<1ms) · Local | Automatic extraction of dates, places, people, and semantics |
| **Embeddings** | **`all-MiniLM-L6-v2`** | ~15ms · CPU / GPU | 384-dimensional dense semantic vector space (Zvec) |
| **Vector Engine** | **Zvec Index** | Instant (<5ms) · RAM/Disk | Fast cosine similarity vector search and retrieval |
| **Face Detection** | **YuNet + SFace (ONNX)** | Real-time · CPU / GPU | 128-dimensional cosine face clustering |
| **Geocoding** | **Reverse Geocoder** | Instant (<1ms) · Local DB | Offline GPS to City/Country mapping |
| **Desktop Shell** | **Tauri v2 (Rust)** | Native binary (<15MB) | Lightweight native window with glass styling & OS bridges |

---

## 🦙 Ollama Setup Guide (End-to-End)

PixelMemory uses [Ollama](https://ollama.com/) for local GPU acceleration of Moondream2 and Gemma 4. Follow the setup steps below for your operating system:

### 🪟 Windows Setup

1. **Install Ollama**:
   - **Option A (One-command with winget)**:
     ```powershell
     winget install -e --id Ollama.Ollama
     ```
   - **Option B (Installer)**:
     Download and run the installer from [ollama.com/download/windows](https://ollama.com/download/windows).

2. **Start Ollama**:
   Launch Ollama from your Start Menu. A llama icon will appear in your Windows System Tray (near the clock).

3. **Pull the Default Models**:
   Open PowerShell or Command Prompt and run:
   ```powershell
   ollama pull moondream
   ollama pull gemma4:e2b
   ```

---

### 🍏 macOS Setup (Apple Silicon M1/M2/M3/M4 & Intel)

1. **Install Ollama**:
   - **Option A (Homebrew)**:
     ```bash
     brew install --cask ollama
     ```
   - **Option B (Direct Download)**:
     Download the `.zip` from [ollama.com/download/mac](https://ollama.com/download/mac) and drag `Ollama.app` into `/Applications`.

2. **Start Ollama**:
   Launch **Ollama** from Applications or Spotlight. An Ollama menu bar icon will appear at the top of your screen.

3. **Pull the Default Models**:
   Open Terminal and run:
   ```bash
   ollama pull moondream
   ollama pull gemma4:e2b
   ```

---

### ✅ Verify Ollama Installation

To confirm both models are ready, run:
```bash
ollama list
```
You should see `moondream:latest` and `gemma4:e2b` listed.

---

## ⚡ Frictionless Installation

### 🪟 Windows Quick Start

PixelMemory provides automated setup scripts that configure the Python virtual environment and check your tools:

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/xklabs-AI/PixelMemory.git
   cd PixelMemory
   ```

2. **Run the automated setup**:
   Double-click `setup.bat` or run in PowerShell:
   ```powershell
   .\setup.bat
   ```
   *(Or using PowerShell: `.\setup.ps1`)*

   > **Zero-Friction Tip**: If you simply run `.\launch.bat` on a fresh system, it will automatically detect that setup is needed and configure `.venv` for you!

---

### 🍏 macOS & Linux Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xklabs-AI/PixelMemory.git
   cd PixelMemory
   ```

2. **Run the automated setup**:
   ```bash
   chmod +x setup.sh launch.sh
   ./setup.sh
   ```

---

## 🚀 Running PixelMemory

PixelMemory can run either as a **Native Desktop Application** or as a **Local Web Platform**.

### Option 1: Native Desktop Application (Tauri v2)

The desktop mode provides native OS window dragging, glass styling, and native file dialogs:

- **Windows**:
  ```powershell
  .\launch.bat --desktop
  # or: .\launch.ps1 -desktop
  ```
- **macOS / Linux**:
  ```bash
  ./launch.sh --desktop
  ```

> [!NOTE]
> **Compiling Desktop Mode**:
> Desktop mode requires **Rust/Cargo** ([rustup.rs](https://rustup.rs)) and Microsoft C++ Build Tools on Windows. If not present, PixelMemory will inform you and gracefully offer to run in Web Browser Mode.

---

### Option 2: Web Browser Platform (FastAPI)

Runs the high-speed local server and automatically opens your default web browser:

- **Windows**:
  ```powershell
  .\launch.bat
  ```
- **macOS / Linux**:
  ```bash
  ./launch.sh
  ```

Default URL: 👉 **`http://localhost:8642`**

---

### Option 3: Quick Demo Archive (12 Sample Photos)

Want to try PixelMemory instantly without waiting for your photo library to index?
- **Windows**:
  ```powershell
  .\launch.bat --demo
  ```
- **macOS / Linux**:
  ```bash
  ./launch.sh --demo
  ```
This seeds 12 curated memories (landscapes, birthdays, pets, food, travel) with GPS and embeddings in ~2 seconds so you can test natural language search right away!

---

## 🩺 System Diagnostics & Doctor

To verify your hardware compute, Ollama status, and database health at any time:

```bash
# Check system status
python launch.py --status

# Interactive setup and model doctor
python launch.py --doctor
```

---

## 📥 Ingesting Your Photo Library

You can import photos directly through the UI:
1. Open PixelMemory (Desktop App or Browser).
2. Click **📁 Ingest Photos** or click **Browse Folder...** (which opens your native OS folder chooser).
3. Select your photo directory (e.g., `D:\Photos` or `/Users/name/Pictures`).
4. Click **Start Ingestion**. The floating progress dock will monitor progress in the background while you continue searching!

---

## ⚙️ Configuration Reference

Settings can be customized in [`backend/config.py`](backend/config.py):

| Setting | Default | Description |
| :--- | :--- | :--- |
| `DEFAULT_VLM_MODEL` | `"moondream"` | Primary Vision model for image captioning |
| `STORY_LLM_MODEL` | `"gemma4:e2b"` | Primary LLM for travel story generation |
| `OLLAMA_HOST` | `"http://localhost:11434"` | Local Ollama API server endpoint |
| `DATA_DIR` | `~/.pixelmemory` | Library database, thumbnails, and Zvec vector store |
| `PORT` | `8642` | Local backend port |
| `HOST` | `"0.0.0.0"` | Network bind address |

---

## 🗺️ Roadmap

- [ ] 💬 **Chat with a Photo** — Ask questions about any photo in your library and get natural, conversational answers
- [ ] 👨‍👩‍👧‍👦 **Chat with a Family Member** — Select a tagged person and discover what they love, where they've been, and who they spend time with — all from your photo history
- [ ] 🔍 **Duplicate Photo Detection** — Find and manage duplicate or near-duplicate photos across your library
- [ ] 📦 **Standalone Windows Installer** — One-click setup, no Python or Rust required
- [ ] 🎞️ **Slideshow Generator** — Turn a selection of photos into an animated slideshow with music

---

## 🛡️ Privacy Commitment

- **Zero Cloud**: 100% of image bytes, facial recognition embeddings, and EXIF coordinates remain strictly on your local disk.
- **No Telemetry**: No tracking cookies, analytics pings, or cloud API calls.
- **Gitignored Libraries**: Your photos, database, and thumbnails are excluded from version control by default.

---

## 📄 License

PixelMemory is open-source software licensed under the [Apache 2.0 License](LICENSE).
