"""Facial and Pet Recognition Engine — YuNet detection and SFace 128-d recognition."""

import os
import io
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

from backend.config import DATA_DIR

# ── Models Directory Resolution ───────────────────────
BUNDLED_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
USER_MODELS_DIR = DATA_DIR / "models"
FACES_THUMB_DIR = DATA_DIR / "face_thumbs"

YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
NANODET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx"
)


def _resolve_model_path(filename: str, min_size: int = 10000) -> Path:
    """Find model in bundled repo models/ directory or user ~/.pixelmemory/models directory."""
    bundled = BUNDLED_MODELS_DIR / filename
    if bundled.exists() and bundled.stat().st_size >= min_size:
        return bundled
    user_p = USER_MODELS_DIR / filename
    if user_p.exists() and user_p.stat().st_size >= min_size:
        return user_p
    return user_p


YUNET_PATH = _resolve_model_path("face_detection_yunet_2023mar.onnx", 100000)
SFACE_PATH = _resolve_model_path("face_recognition_sface_2021dec.onnx", 10000000)
NANODET_PATH = _resolve_model_path("object_detection_nanodet_2022nov.onnx", 1000000)

# SFace cosine similarity threshold (0.363 is official threshold for high confidence match)
SIMILARITY_THRESHOLD = 0.363

_detector = None
_recognizer = None
_nanodet = None

# NanoDet precomputed anchors and normalization
_NANODET_STRIDES = (8, 16, 32)
_NANODET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
_NANODET_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
_NANODET_PROJECT = np.arange(8)
_NANODET_ANCHORS = []
for _stride in _NANODET_STRIDES:
    _feat_w = int(416 / _stride)
    _feat_h = int(416 / _stride)
    _shift_x = np.arange(0, _feat_w) * _stride
    _shift_y = np.arange(0, _feat_h) * _stride
    _xv, _yv = np.meshgrid(_shift_x, _shift_y)
    _cx = _xv.flatten() + 0.5 * (_stride - 1)
    _cy = _yv.flatten() + 0.5 * (_stride - 1)
    _NANODET_ANCHORS.append(np.column_stack((_cx, _cy)))


def ensure_models() -> bool:
    """Ensure YuNet, SFace, and NanoDet models are available from local bundle or download fallback."""
    global YUNET_PATH, SFACE_PATH, NANODET_PATH
    USER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FACES_THUMB_DIR.mkdir(parents=True, exist_ok=True)

    YUNET_PATH = _resolve_model_path("face_detection_yunet_2023mar.onnx", 100000)
    SFACE_PATH = _resolve_model_path("face_recognition_sface_2021dec.onnx", 10000000)
    NANODET_PATH = _resolve_model_path("object_detection_nanodet_2022nov.onnx", 1000000)

    # Check and download YuNet if needed
    if not YUNET_PATH.exists() or YUNET_PATH.stat().st_size < 100000:
        target = USER_MODELS_DIR / "face_detection_yunet_2023mar.onnx"
        print("[Faces] Downloading YuNet face detection model...")
        try:
            urllib.request.urlretrieve(YUNET_URL, target)
            YUNET_PATH = target
        except Exception as e:
            print(f"[Faces] Error downloading YuNet: {e}")

    # Check and download SFace if needed
    if not SFACE_PATH.exists() or SFACE_PATH.stat().st_size < 10000000:
        target = USER_MODELS_DIR / "face_recognition_sface_2021dec.onnx"
        print("[Faces] Downloading SFace face recognition model...")
        try:
            urllib.request.urlretrieve(SFACE_URL, target)
            SFACE_PATH = target
        except Exception as e:
            print(f"[Faces] Error downloading SFace: {e}")

    # Check and download NanoDet if needed
    if not NANODET_PATH.exists() or NANODET_PATH.stat().st_size < 1000000:
        target = USER_MODELS_DIR / "object_detection_nanodet_2022nov.onnx"
        print("[Faces] Downloading NanoDet pet detection model...")
        try:
            urllib.request.urlretrieve(NANODET_URL, target)
            NANODET_PATH = target
        except Exception as e:
            print(f"[Faces] Error downloading NanoDet: {e}")

    return YUNET_PATH.exists() and SFACE_PATH.exists() and NANODET_PATH.exists()


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


def get_nanodet():
    """Get or initialize NanoDet object detector for pets."""
    global _nanodet
    ensure_models()
    if _nanodet is None and NANODET_PATH.exists():
        _nanodet = cv2.dnn.readNet(str(NANODET_PATH))
    return _nanodet


def detect_pets_nanodet(bgr: np.ndarray, orig_w: int, orig_h: int, score_threshold: float = 0.20) -> list[dict]:
    """
    Detect cats and dogs using lightweight NanoDet (COCO classes 15=cat, 16=dog, 14=bird).
    Runs letterboxed 416x416 inference and returns normalized boxes with SFace embeddings.
    """
    net = get_nanodet()
    if net is None:
        return []

    target_size = (416, 416)
    hw_scale = orig_h / orig_w
    if hw_scale > 1:
        newh, neww = target_size[0], int(target_size[1] / hw_scale)
        resized = cv2.resize(bgr, (neww, newh), interpolation=cv2.INTER_AREA)
        left = int((target_size[1] - neww) * 0.5)
        padded = cv2.copyMakeBorder(resized, 0, 0, left, target_size[1] - neww - left, cv2.BORDER_CONSTANT, value=0)
        scale_info = (0, left, newh, neww)
    else:
        newh, neww = int(target_size[0] * hw_scale), target_size[1]
        resized = cv2.resize(bgr, (neww, newh), interpolation=cv2.INTER_AREA)
        top = int((target_size[0] - newh) * 0.5)
        padded = cv2.copyMakeBorder(resized, top, target_size[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        scale_info = (top, 0, newh, neww)

    blob = (padded.astype(np.float32) - _NANODET_MEAN) / _NANODET_STD
    blob = cv2.dnn.blobFromImage(blob)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    cls_scores, bbox_preds = outs[:3], outs[3:]

    # Classes: 15 (cat), 16 (dog), 14 (bird)
    PET_CLASSES = {15: "cat", 16: "dog", 14: "bird"}
    all_bboxes, all_scores, all_classes = [], [], []

    for stride, cls_score, bbox_pred, anchors in zip(_NANODET_STRIDES, cls_scores, bbox_preds, _NANODET_ANCHORS):
        cls_score = cls_score.squeeze(0)
        bbox_pred = bbox_pred.squeeze(0)
        x_exp = np.exp(bbox_pred.reshape(-1, 8))
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        bbox_pred = (x_exp / x_sum).dot(_NANODET_PROJECT).reshape(-1, 4) * stride
        x1 = anchors[:, 0] - bbox_pred[:, 0]
        y1 = anchors[:, 1] - bbox_pred[:, 1]
        x2 = anchors[:, 0] + bbox_pred[:, 2]
        y2 = anchors[:, 1] + bbox_pred[:, 3]
        boxes = np.column_stack([x1, y1, x2, y2])
        for cid in PET_CLASSES:
            scores = cls_score[:, cid]
            mask = scores > score_threshold
            if np.any(mask):
                all_bboxes.append(boxes[mask])
                all_scores.append(scores[mask])
                all_classes.append(np.full(np.sum(mask), cid))

    if not all_bboxes:
        return []

    all_bboxes = np.vstack(all_bboxes)
    all_scores = np.concatenate(all_scores)
    all_classes = np.concatenate(all_classes)

    cv_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in all_bboxes]
    indices = cv2.dnn.NMSBoxes(cv_boxes, all_scores.tolist(), score_threshold, 0.25)

    recognizer = get_recognizer()
    results = []
    top, left, nh, nw = scale_info

    for idx in indices:
        cid = int(all_classes[idx])
        conf = float(all_scores[idx])
        box = all_bboxes[idx]
        rx1 = max(0.0, min(1.0, (box[0] - left) / nw))
        ry1 = max(0.0, min(1.0, (box[1] - top) / nh))
        rx2 = max(0.0, min(1.0, (box[2] - left) / nw))
        ry2 = max(0.0, min(1.0, (box[3] - top) / nh))
        rw = max(0.01, min(1.0 - rx1, rx2 - rx1))
        rh = max(0.01, min(1.0 - ry1, ry2 - ry1))

        px1, py1 = int(rx1 * orig_w), int(ry1 * orig_h)
        px2, py2 = int((rx1 + rw) * orig_w), int((ry1 + rh) * orig_h)
        crop = bgr[py1:py2, px1:px2]
        feat_bytes = None
        if crop.size > 0:
            try:
                resized = cv2.resize(crop, (112, 112))
                feat = recognizer.feature(resized)
                feat_bytes = feat.astype(np.float32).tobytes()
            except Exception:
                pass

        results.append({
            "box_x": round(rx1, 4),
            "box_y": round(ry1, 4),
            "box_w": round(rw, 4),
            "box_h": round(rh, 4),
            "confidence": round(conf, 3),
            "embedding": feat_bytes,
            "is_pet": True,
            "pet_label": PET_CLASSES.get(cid, "pet"),
        })

    return results


def detect_and_embed_faces(image_path: str | Path, score_threshold: float = 0.45) -> list[dict]:
    """
    Run YuNet detection for human faces + NanoDet for pets (cats & dogs) + SFace embedding extraction.
    Applies EXIF orientation transpose so coordinates match browser orientation.
    Returns list of dicts:
      {
        'box_x': float (0.0..1.0),
        'box_y': float (0.0..1.0),
        'box_w': float (0.0..1.0),
        'box_h': float (0.0..1.0),
        'confidence': float,
        'embedding': bytes (128 float32),
        'is_pet': bool,
        'pet_label': str | None,
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

    # 1. Run YuNet human face detection
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
    results = []
    if faces is not None and len(faces) > 0:
        for face in faces:
            score = float(face[-1])
            scaled_face = face.copy()
            scaled_face[0:4] = face[0:4] / scale
            if len(face) > 4:
                scaled_face[4:14] = face[4:14] / scale

            x, y, fw, fh = scaled_face[0:4]
            norm_x = max(0.0, min(1.0, float(x) / w))
            norm_y = max(0.0, min(1.0, float(y) / h))
            norm_w = max(0.0, min(1.0 - norm_x, float(fw) / w))
            norm_h = max(0.0, min(1.0 - norm_y, float(fh) / h))

            try:
                aligned = recognizer.alignCrop(input_bgr, face)
                feat = recognizer.feature(aligned)
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
                "is_pet": False,
                "pet_label": None,
            })

    # 2. Run NanoDet pet detection (cats, dogs, birds)
    try:
        pets = detect_pets_nanodet(bgr, w, h, score_threshold=0.20)
        for p in pets:
            # Prevent pet box from colliding with a detected human face
            overlap = False
            for r in results:
                ix1 = max(p["box_x"], r["box_x"])
                iy1 = max(p["box_y"], r["box_y"])
                ix2 = min(p["box_x"] + p["box_w"], r["box_x"] + r["box_w"])
                iy2 = min(p["box_y"] + p["box_h"], r["box_y"] + r["box_h"])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_p = p["box_w"] * p["box_h"]
                    if area_p > 0 and (inter / area_p) > 0.4:
                        overlap = True
                        break
            if not overlap:
                results.append(p)
    except Exception as e:
        print(f"[Faces] Pet detection error: {e}")

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
