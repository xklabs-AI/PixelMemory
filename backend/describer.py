"""Generate image descriptions using Vision-Language Models (Moondream2, LLaVA 7B, etc.) via local Ollama or PyTorch."""

import json
import base64
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from PIL import Image, ImageOps
from tqdm import tqdm

from backend.config import (
    USE_OLLAMA, OLLAMA_HOST, DEFAULT_VLM_MODEL,
    VLM_MODEL, VLM_REVISION, VLM_PROMPT, AVAILABLE_VLM_MODELS,
)

# Runtime mutable active VLM model (defaults to config DEFAULT_VLM_MODEL)
_ACTIVE_VLM_MODEL = DEFAULT_VLM_MODEL


def get_active_vlm_model() -> str:
    """Return current runtime active VLM model ID."""
    global _ACTIVE_VLM_MODEL
    return _ACTIVE_VLM_MODEL


def set_active_vlm_model(model_id: str) -> str:
    """Set runtime active VLM model ID."""
    global _ACTIVE_VLM_MODEL
    clean_id = model_id.strip()
    if clean_id:
        _ACTIVE_VLM_MODEL = clean_id
    return _ACTIVE_VLM_MODEL


def get_installed_ollama_models(host: str = OLLAMA_HOST) -> list[str]:
    """Return all model tags installed in local Ollama instance."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        pass
    return []


def is_ollama_ready(host: str = OLLAMA_HOST, model_name: Optional[str] = None) -> bool:
    """Check if Ollama server is running and optionally has the target model."""
    target = model_name or get_active_vlm_model()
    installed = get_installed_ollama_models(host)
    if not installed:
        return False
    if not target:
        return True
    return any(target in m or m.startswith(target.split(":")[0]) for m in installed)


def get_available_vlm_models(host: str = OLLAMA_HOST) -> list[dict]:
    """Return list of supported VLM models with real-time Ollama installation status."""
    installed_tags = get_installed_ollama_models(host)
    active = get_active_vlm_model()

    models = []
    for m in AVAILABLE_VLM_MODELS:
        m_id = m["id"]
        is_inst = any(m_id in tag or tag.startswith(m_id.split(":")[0]) for tag in installed_tags)
        models.append({
            "id": m_id,
            "name": m["name"],
            "tag": m["tag"],
            "vram": m["vram"],
            "description": m["description"],
            "installed": is_inst,
            "active": (m_id == active),
        })
    return models


def clean_vlm_text(text: str) -> str:
    """Strip negative checklist boilerplate (e.g. 'does not contain people') to prevent false positive embeddings."""
    import re
    cleaned = re.sub(
        r"(?:The image|It|There)\s+(?:does not|doesn't|contains? no|has no)\s+[^.]*\.?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:There are no|No visible|No discernible)\s+[^.]*\.?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split()).strip()


class ImageDescriber:
    """Wraps Vision-Language Models (Moondream2, LLaVA 7B, etc.) for local image captioning."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or get_active_vlm_model()
        self.use_ollama = False
        self.hf_model = None

    def load(self) -> None:
        """Initialize connection to Ollama or load HuggingFace PyTorch model."""
        if USE_OLLAMA and is_ollama_ready(OLLAMA_HOST, self.model_name):
            print(f"Connected to Ollama at {OLLAMA_HOST} using '{self.model_name}' (GPU Accelerated).")
            self.use_ollama = True
            return

        # Fallback to direct HuggingFace model
        print(f"Ollama not found or '{self.model_name}' not pulled. Falling back to local PyTorch...")
        import torch
        import moondream as md

        print(f"Loading {VLM_MODEL} (revision: {VLM_REVISION})...")
        self.hf_model = md.vl(model=VLM_MODEL, revision=VLM_REVISION)
        print("Model loaded on GPU." if torch.cuda.is_available() else "Model loaded on CPU (slow).")

    def _describe_ollama(self, file_path: str) -> str:
        """Describe image using local Ollama instance."""
        p = Path(file_path)
        if not p.exists():
            return f"[file not found: {file_path}]"

        try:
            import io
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Downscale large camera photos to max 1024x1024 for fast inference
                img.thumbnail((1024, 1024))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            return f"[image open failed: {e}]"

        payload = {
            "model": self.model_name,
            "prompt": VLM_PROMPT,
            "images": [b64_data],
            "stream": False,
        }

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()

    def _describe_hf(self, file_path: str) -> str:
        """Describe image using direct HuggingFace model."""
        with Image.open(file_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            encoded = self.hf_model.encode_image(img)
            result = self.hf_model.query(encoded, VLM_PROMPT)["answer"]
            return result.strip()

    def describe(self, file_path: str) -> str:
        """Generate a description for a single image."""
        if not self.use_ollama and self.hf_model is None:
            self.load()

        try:
            if self.use_ollama:
                raw = self._describe_ollama(file_path)
            else:
                raw = self._describe_hf(file_path)
            return clean_vlm_text(raw)
        except Exception as e:
            return f"[description failed: {e}]"

    def describe_batch(self, items: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """
        Describe a batch of (image_id, file_path) pairs.
        Returns list of (image_id, description).
        """
        if not self.use_ollama and self.hf_model is None:
            self.load()

        results = []
        for image_id, fpath in tqdm(items, desc=f"Describing images ({self.model_name})", unit="img"):
            desc = self.describe(fpath)
            results.append((image_id, desc))
        return results


def enrich_description(raw_description: str, metadata: dict) -> str:
    """
    Prepend date and location context to the VLM description.
    This is the key insight — the enriched text is what gets embedded,
    so semantic search on "Grand Canyon 2016" hits the right images.
    """
    parts = []

    # Date context
    date_taken = metadata.get("date_taken")
    if date_taken:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(date_taken)
            parts.append(dt.strftime("%B %Y"))  # e.g. "March 2016"
        except (ValueError, TypeError):
            pass

    # Location context
    place_name = metadata.get("place_name")
    if place_name:
        parts.append(place_name)

    # Camera context
    camera = metadata.get("camera_model")
    if camera:
        parts.append(f"taken with {camera}")

    prefix = ", ".join(parts)
    if prefix:
        return f"{prefix} — {raw_description}"
    return raw_description
