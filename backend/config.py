"""PixelMemory configuration."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────
DATA_DIR = Path.home() / ".pixelmemory"
DB_PATH = DATA_DIR / "pixelmemory.db"
CHROMA_DIR = DATA_DIR / "chroma"  # kept for migration script
ZVEC_DIR = DATA_DIR / "zvec"
ZVEC_DIMENSION = 384              # all-MiniLM-L6-v2 output dimensionality
THUMB_DIR = DATA_DIR / "thumbnails"
THUMB_SIZE = (320, 320)

# ── Models ───────────────────────────────────────────
USE_OLLAMA = True
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_VLM_MODEL = "moondream:1.8b"
ACTIVE_VLM_MODEL = "moondream:1.8b"
OLLAMA_MODEL = "moondream:1.8b"        # fallback / legacy reference
VLM_MODEL = "vikhyatk/moondream2"      # HuggingFace fallback
VLM_REVISION = "2025-01-09"           # pin for reproducibility (HuggingFace fallback)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

AVAILABLE_VLM_MODELS = [
    {
        "id": "moondream:1.8b",
        "name": "Moondream2 (1.8B)",
        "tag": "⚡ Fast · 1.8GB VRAM",
        "vram": "1.8 GB",
        "description": "Ultra-fast local captioning (~0.8s/img), great for high-volume photo indexing.",
    },
    {
        "id": "llava:7b",
        "name": "LLaVA (7B)",
        "tag": "🧠 Deep Detail · 4.5GB VRAM",
        "vram": "4.5 GB",
        "description": "Higher compute & deeper visual reasoning (~2-3s/img), highly nuanced scene descriptions.",
    },
]

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

# ── Story Timeline ──────────────────────────────────
STORY_LLM_MODEL = "gemma4:e4b"

AVAILABLE_STORY_MODELS = [
    {
        "id": "gemma4:e4b",
        "name": "Gemma 4 (E4B)",
        "tag": "🧠 Rich Prose · ~4B",
        "description": "Higher quality narratives, more vivid storytelling.",
    },
    {
        "id": "gemma4:e2b",
        "name": "Gemma 4 (E2B)",
        "tag": "⚡ Fast · ~2B",
        "description": "Faster generation, great for quick story previews.",
    },
]
STORY_PROMPT_TEMPLATE = (
    "You are a warm, personal travel journal writer. Based on these photo descriptions "
    "from {date}, write a short first-person narrative paragraph (3-5 sentences) "
    "summarizing the day. Write naturally as if fondly recalling the day's highlights. "
    "Include specific places, foods, and activities mentioned. Do not list the photos. "
    "Do not mention photo numbers or that you are looking at photos. Just tell the story of the day."
)
