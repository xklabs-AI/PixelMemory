"""Extract EXIF metadata and reverse-geocode GPS coordinates."""

from datetime import datetime
from pathlib import Path

import exifread
import reverse_geocoder as rg
from PIL import Image

from backend.config import THUMB_DIR, THUMB_SIZE


def _to_decimal(tag_value) -> float | None:
    """Convert EXIF GPS DMS (degrees/minutes/seconds) to decimal."""
    try:
        values = tag_value.values
        d = float(values[0].num) / float(values[0].den)
        m = float(values[1].num) / float(values[1].den)
        s = float(values[2].num) / float(values[2].den)
        return d + m / 60 + s / 3600
    except (AttributeError, IndexError, ZeroDivisionError):
        return None


def extract_exif(file_path: str) -> dict:
    """
    Extract date, GPS, camera info from EXIF.
    Returns a dict of fields matching db.update_metadata kwargs.
    """
    meta = {}

    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return meta

    # ── Date ─────────────────────────────────────────
    for date_tag in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
        raw = tags.get(date_tag)
        if raw:
            try:
                dt = datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
                meta["date_taken"] = dt.isoformat()
            except ValueError:
                pass
            break

    # ── GPS ──────────────────────────────────────────
    lat_tag = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon_tag = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")

    if lat_tag and lon_tag:
        lat = _to_decimal(lat_tag)
        lon = _to_decimal(lon_tag)
        if lat is not None and lon is not None:
            if lat_ref and str(lat_ref) == "S":
                lat = -lat
            if lon_ref and str(lon_ref) == "W":
                lon = -lon
            meta["latitude"] = lat
            meta["longitude"] = lon

    # ── Camera ───────────────────────────────────────
    make = tags.get("Image Make")
    model = tags.get("Image Model")
    if make:
        meta["camera_make"] = str(make).strip()
    if model:
        meta["camera_model"] = str(model).strip()

    orientation = tags.get("Image Orientation")
    if orientation:
        try:
            meta["orientation"] = int(str(orientation))
        except ValueError:
            pass

    return meta


def reverse_geocode(lat: float, lon: float) -> dict:
    """
    Offline reverse geocode using the reverse_geocoder library.
    Returns {city, region, country, place_name}.
    """
    try:
        result = rg.search((lat, lon), mode=1)  # mode=1 = single-threaded, fast
        if result:
            r = result[0]
            city = r.get("name", "")
            region = r.get("admin1", "")
            country = r.get("cc", "")
            # Build the most useful place name string
            parts = [p for p in (city, region, country) if p]
            return {
                "city": city,
                "region": region,
                "country": country,
                "place_name": ", ".join(parts),
            }
    except Exception:
        pass
    return {}


def generate_thumbnail(file_path: str, image_id: int) -> Path | None:
    """Create a JPEG thumbnail for the search UI."""
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMB_DIR / f"{image_id}.jpg"
    if thumb_path.exists():
        return thumb_path

    try:
        with Image.open(file_path) as img:
            img.thumbnail(THUMB_SIZE)
            # Convert to RGB if needed (handles RGBA, palette, etc.)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(thumb_path, "JPEG", quality=80)
        return thumb_path
    except Exception:
        return None


def process_metadata(image_id: int, file_path: str) -> dict:
    """
    Full CPU-side processing for one image:
    extract EXIF → reverse geocode → generate thumbnail.
    Returns the metadata dict.
    """
    meta = extract_exif(file_path)

    # Reverse geocode if we have GPS
    if "latitude" in meta and "longitude" in meta:
        geo = reverse_geocode(meta["latitude"], meta["longitude"])
        meta.update(geo)

    # Thumbnail
    generate_thumbnail(file_path, image_id)

    return meta
