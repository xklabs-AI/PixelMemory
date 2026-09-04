"""Generate image descriptions using Moondream on local GPU (via Ollama or direct PyTorch)."""

import json
import base64
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

from backend.config import (
    USE_OLLAMA, OLLAMA_HOST, OLLAMA_MODEL,
    VLM_MODEL, VLM_REVISION, VLM_PROMPT,
)


def is_ollama_ready(host: str = OLLAMA_HOST, model_name: str = OLLAMA_MODEL) -> bool:
    """Check if Ollama server is running and has the target model."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # Check for exact or prefix match (e.g. moondream:1.8b)
                return any(model_name in m or m.startswith(model_name.split(":")[0]) for m in models)
    except Exception:
        pass
    return False


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
    """Wraps Moondream (via Ollama or direct HuggingFace) for image captioning."""

    def __init__(self):
        self.use_ollama = False
        self.hf_model = None

    def load(self) -> None:
        """Initialize connection to Ollama or load HuggingFace PyTorch model."""
        if USE_OLLAMA and is_ollama_ready(OLLAMA_HOST, OLLAMA_MODEL):
            print(f"Connected to Ollama at {OLLAMA_HOST} using '{OLLAMA_MODEL}' (GPU Accelerated).")
            self.use_ollama = True
            return

        # Fallback to direct HuggingFace model
        print(f"Ollama not found or '{OLLAMA_MODEL}' not pulled. Falling back to local PyTorch...")
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
            "model": OLLAMA_MODEL,
            "prompt": VLM_PROMPT,
            "images": [b64_data],
            "stream": False,
        }

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=90) as resp:
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
        for image_id, fpath in tqdm(items, desc="Describing images (VLM)", unit="img"):
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
