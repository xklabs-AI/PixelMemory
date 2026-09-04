"""PixelMemory configuration."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────
DATA_DIR = Path.home() / ".pixelmemory"
DB_PATH = DATA_DIR / "pixelmemory.db"
CHROMA_DIR = DATA_DIR / "chroma"
THUMB_DIR = DATA_DIR / "thumbnails"
THUMB_SIZE = (320, 320)

# ── Models ───────────────────────────────────────────
USE_OLLAMA = True
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "moondream:1.8b"
VLM_MODEL = "vikhyatk/moondream2"
VLM_REVISION = "2025-01-09"           # pin for reproducibility (HuggingFace fallback)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Ingest ───────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".tif", ".tiff", ".bmp",
}
HASH_CHUNK_SIZE = 1 << 16             # 64 KB read chunks for hashing
BATCH_SIZE = 16                        # images per GPU batch
CPU_WORKERS = 4                        # metadata extraction threads
VLM_PROMPT = (
    "Describe this photograph in comprehensive detail for semantic search. "
    "Identify the broad category and scene type (e.g. food & cuisine, animals & wildlife, vehicles & transport, people & portraits, nature & landscape, architecture & interior). "
    "State the primary subjects, their actions, and clothing; "
    "all visible objects, food items, animals, or vehicles; "
    "the environment, setting, terrain, weather, background, prominent colors, lighting, and mood. "
    "Only describe what is visible; never describe what is absent or missing."
)

# ── Server ───────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8642
SEARCH_TOP_K = 40
