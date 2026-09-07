"""Facial and Pet Recognition Engine — YuNet detection and SFace 128-d recognition."""

import os
import io
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

from backend.config import DATA_DIR

MODELS_DIR = DATA_DIR / "models"
FACES_THUMB_DIR = DATA_DIR / "face_thumbs"

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# SFace cosine similarity threshold (0.363 is official threshold for high confidence match)
SIMILARITY_THRESHOLD = 0.363

_detector = None
_recognizer = None


def ensure_models() -> bool:
    """Download YuNet and SFace models if not already present."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FACES_THUMB_DIR.mkdir(parents=True, exist_ok=True)

    # Check and download YuNet if needed
    if not YUNET_PATH.exists() or YUNET_PATH.stat().st_size < 100000:
        YUNET_PATH.unlink(missing_ok=True)
        print(f"[Faces] Downloading YuNet face detection model...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)

    # Check and download SFace if needed
    if not SFACE_PATH.exists() or SFACE_PATH.stat().st_size < 10000000:
        SFACE_PATH.unlink(missing_ok=True)
        print(f"[Faces] Downloading SFace face recognition model...")
        urllib.request.urlretrieve(SFACE_URL, SFACE_PATH)

    return YUNET_PATH.exists() and SFACE_PATH.exists()


def get_detector(width: int, height: int):
    """Get or initialize YuNet face detector with specified input dimensions."""
    global _detector
    ensure_models()
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            str(YUNET_PATH),
            "",
            (width, height),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=50,
        )
    else:
        _detector.setInputSize((width, height))
    return _detector


def get_recognizer():
    """Get or initialize SFace face recognizer."""
    global _recognizer
    ensure_models()
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return _recognizer


def detect_and_embed_faces(image_path: str | Path, score_threshold: float = 0.45) -> list[dict]:
    """
    Run YuNet detection + SFace embedding extraction on an image.
    Applies EXIF orientation transpose so coordinates match browser orientation.
    Returns list of dicts:
      {
        'box_x': float (0.0..1.0),
        'box_y': float (0.0..1.0),
        'box_w': float (0.0..1.0),
        'box_h': float (0.0..1.0),
        'confidence': float,
        'embedding': bytes (128 float32),
      }
    """
    ensure_models()
    p = str(image_path)
    if not os.path.exists(p):
        return []

    try:
        with Image.open(p) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            w, h = pil_img.size
            if w == 0 or h == 0:
                return []
            rgb = np.array(pil_img.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[Faces] Error opening image {p}: {e}")
        return []

    # Scale image for optimal face detection receptive field
    max_dim = 1600
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        dw, dh = int(w * scale), int(h * scale)
        input_bgr = cv2.resize(bgr, (dw, dh))
    else:
        dw, dh = w, h
        input_bgr = bgr

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (dw, dh),
        score_threshold=score_threshold,
        nms_threshold=0.3,
        top_k=50,
    )
    recognizer = get_recognizer()

    _, faces = detector.detect(input_bgr)
    if faces is None or len(faces) == 0:
        return []

    results = []
    for face in faces:
        score = float(face[-1])
        # Rescale face box and landmarks back to original image size
        scaled_face = face.copy()
        scaled_face[0:4] = face[0:4] / scale
        if len(face) > 4:
            scaled_face[4:14] = face[4:14] / scale

        x, y, fw, fh = scaled_face[0:4]

        # Clamp normalized coordinates between 0.0 and 1.0
        norm_x = max(0.0, min(1.0, float(x) / w))
        norm_y = max(0.0, min(1.0, float(y) / h))
        norm_w = max(0.0, min(1.0 - norm_x, float(fw) / w))
        norm_h = max(0.0, min(1.0 - norm_y, float(fh) / h))

        # Align crop from input_bgr and extract 128-d embedding
        try:
            aligned = recognizer.alignCrop(input_bgr, face)
            feat = recognizer.feature(aligned)  # shape (1, 128) float32
            feat_bytes = feat.astype(np.float32).tobytes()
        except Exception:
            feat_bytes = None

        results.append({
            "box_x": round(norm_x, 4),
            "box_y": round(norm_y, 4),
            "box_w": round(norm_w, 4),
            "box_h": round(norm_h, 4),
            "confidence": round(score, 3),
            "embedding": feat_bytes,
        })

    return results


def match_face_embedding(
    target_embedding_bytes: bytes,
    known_embeddings: list[tuple[int, int, bytes]],  # [(face_id, person_id, bytes)]
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[int | None, float]:
    """
    Compare target embedding against known person embeddings using cosine similarity.
    Returns (matched_person_id, max_score) or (None, 0.0).
    """
    if not target_embedding_bytes or not known_embeddings:
        return None, 0.0

    target_feat = np.frombuffer(target_embedding_bytes, dtype=np.float32).reshape(1, -1)
    recognizer = get_recognizer()

    best_person_id = None
    best_score = -1.0

    # Group scores by person to find closest match
    person_scores: dict[int, list[float]] = {}
    for _, person_id, emb_bytes in known_embeddings:
        if not emb_bytes or person_id is None:
            continue
        feat = np.frombuffer(emb_bytes, dtype=np.float32).reshape(1, -1)
        score = float(recognizer.match(target_feat, feat, cv2.FaceRecognizerSF_FR_COSINE))
        person_scores.setdefault(person_id, []).append(score)

    for pid, scores in person_scores.items():
        max_s = max(scores)
        if max_s > best_score:
            best_score = max_s
            best_person_id = pid

    if best_score >= threshold:
        return best_person_id, round(best_score, 3)

    return None, round(max(0.0, best_score), 3)


def crop_face_thumbnail(image_path: str | Path, box: dict, output_size: int = 160) -> bytes | None:
    """
    Crop face with slight aesthetic padding, resize to square, and return JPEG bytes.
    box has keys 'box_x', 'box_y', 'box_w', 'box_h' (all 0.0..1.0).
    """
    p = str(image_path)
    if not os.path.exists(p):
        return None

    try:
        with Image.open(p) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size

            bx = box["box_x"] * w
            by = box["box_y"] * h
            bw = box["box_w"] * w
            bh = box["box_h"] * h

            # Add 25% padding around the face for context
            pad_x = bw * 0.25
            pad_y = bh * 0.25

            left = max(0, int(bx - pad_x))
            top = max(0, int(by - pad_y))
            right = min(w, int(bx + bw + pad_x))
            bottom = min(h, int(by + bh + pad_y))

            if right <= left or bottom <= top:
                return None

            crop = img.crop((left, top, right, bottom))
            crop = crop.resize((output_size, output_size), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=88)
            return buf.getvalue()
    except Exception as e:
        print(f"[Faces] Error cropping thumbnail: {e}")
        return None
