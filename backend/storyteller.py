"""Story Timeline — group album photos by day and generate LLM narratives."""

import json
import urllib.request
import urllib.error
from collections import OrderedDict
from datetime import datetime

from backend.config import OLLAMA_HOST, STORY_LLM_MODEL, STORY_PROMPT_TEMPLATE, GROUP_STORY_PROMPT_TEMPLATE
from backend.db import (
    get_conn, get_album_image_ids, get_image_by_id,
    get_cached_narrative, upsert_narrative,
)


def group_photos_by_day(
    photos: list[dict],
    start_date: str | None = None,
    end_date: str | None = None,
) -> OrderedDict:
    """
    Group photos by calendar date, filtering to optional date range.
    Returns OrderedDict {day_str: [photo_dicts]} sorted chronologically.
    Photos without dates are grouped under 'undated'.
    """
    buckets: dict[str, list[dict]] = {}

    for p in photos:
        dt_str = p.get("date_taken") or ""
        if not dt_str:
            buckets.setdefault("undated", []).append(p)
            continue

        day = dt_str[:10]  # 'YYYY-MM-DD'

        # Apply date range filter
        if start_date and day < start_date:
            continue
        if end_date and day > end_date:
            continue

        buckets.setdefault(day, []).append(p)

    # Sort each bucket by time within the day
    for day, day_photos in buckets.items():
        if day != "undated":
            day_photos.sort(key=lambda p: p.get("date_taken") or "")

    # Build ordered result: dated days first (chronological), then undated
    ordered = OrderedDict()
    for day in sorted(k for k in buckets if k != "undated"):
        ordered[day] = buckets[day]
    if "undated" in buckets:
        ordered["undated"] = buckets["undated"]

    return ordered


def build_day_prompt(day_date: str, photos: list[dict]) -> str:
    """
    Construct the LLM prompt for a single day's photos.
    Includes timestamps, locations, and VLM descriptions.
    """
    if day_date == "undated":
        date_label = "an unknown date"
    else:
        try:
            dt = datetime.strptime(day_date, "%Y-%m-%d")
            date_label = dt.strftime("%A, %B %d, %Y")  # e.g. "Saturday, July 24, 2016"
        except ValueError:
            date_label = day_date

    header = STORY_PROMPT_TEMPLATE.format(date=date_label)
    lines = [header, "", f"Photos from {date_label}:", ""]

    for i, p in enumerate(photos, 1):
        dt_str = p.get("date_taken") or ""
        time_part = ""
        if dt_str and "T" in dt_str:
            try:
                dt = datetime.fromisoformat(dt_str)
                time_part = dt.strftime("%I:%M %p")
            except ValueError:
                pass

        place = p.get("place_name") or ""
        desc = p.get("raw_description") or p.get("enriched_text") or "(no description)"

        location_info = f", {place}" if place else ""
        time_info = f"{time_part}" if time_part else "unknown time"

        lines.append(f"Photo {i} ({time_info}{location_info}): \"{desc}\"")

    return "\n".join(lines)


def build_group_prompt(group_title: str, photos: list[dict]) -> str:
    """
    Construct the LLM prompt for an arbitrary group/section of photos.
    Handles date groups, location groups, camera groups, or custom selections.
    """
    title_label = group_title.strip() if group_title else "Photo Group"
    header = GROUP_STORY_PROMPT_TEMPLATE.format(section_title=title_label)
    lines = [header, "", f"Photos from \"{title_label}\":", ""]

    for i, p in enumerate(photos, 1):
        dt_str = p.get("date_taken") or ""
        time_part = ""
        if dt_str:
            try:
                if "T" in dt_str:
                    dt = datetime.fromisoformat(dt_str)
                    time_part = dt.strftime("%b %d, %Y %I:%M %p")
                else:
                    time_part = dt_str[:10]
            except ValueError:
                time_part = dt_str

        place = p.get("place_name") or ""
        desc = p.get("raw_description") or p.get("enriched_text") or "(no description)"

        location_info = f", {place}" if place else ""
        time_info = f"{time_part}" if time_part else "undated"

        lines.append(f"Photo {i} ({time_info}{location_info}): \"{desc}\"")

    return "\n".join(lines)


def call_ollama_llm(prompt: str, model: str = STORY_LLM_MODEL) -> str:
    """Call Ollama text generation API and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 1200,  # Accommodate reasoning models (e.g. gemma4 thinking tokens) + narrative
        },
    }

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()


def generate_day_narrative(
    day_date: str,
    photos: list[dict],
    model: str = STORY_LLM_MODEL,
) -> str:
    """Build prompt and generate narrative for a single day."""
    prompt = build_day_prompt(day_date, photos)
    return call_ollama_llm(prompt, model)


def generate_group_narrative(
    group_title: str,
    photos: list[dict],
    model: str = STORY_LLM_MODEL,
) -> str:
    """Build prompt and generate narrative for an arbitrary grouped section or selection."""
    prompt = build_group_prompt(group_title, photos)
    return call_ollama_llm(prompt, model)


def fetch_album_photos(album_id: int) -> list[dict]:
    """Load all photos for an album with full metadata."""
    with get_conn() as conn:
        image_ids = get_album_image_ids(conn, album_id)
        photos = []
        for img_id in image_ids:
            row = get_image_by_id(conn, img_id)
            if row:
                photos.append(dict(row))
        return photos


def generate_single_day_story(
    album_id: int,
    day_date: str,
    model: str = STORY_LLM_MODEL,
    force: bool = True,
) -> dict:
    """
    Generate or regenerate a story narrative for a single day/group in an album.
    Updates the story_cache and returns the narrative info.
    """
    photos = fetch_album_photos(album_id)
    day_groups = group_photos_by_day(photos)
    day_photos = day_groups.get(day_date, [])

    if not day_photos:
        # Fallback: check if day_date matches any date prefix
        day_photos = [p for p in photos if (p.get("date_taken") or "")[:10] == day_date]

    if not day_photos:
        raise ValueError(f"No photos found for day '{day_date}' in album {album_id}")

    photo_ids = [p["id"] for p in day_photos]
    photo_ids_json = json.dumps(photo_ids)

    if not force:
        with get_conn() as conn:
            cached = get_cached_narrative(conn, album_id, day_date)
        if cached and cached.get("model_used") == model and cached.get("photo_ids") == photo_ids_json:
            return {
                "day_date": day_date,
                "narrative": cached["narrative"],
                "photo_ids": photo_ids,
                "photo_count": len(day_photos),
                "model_used": cached.get("model_used", model),
                "cached": True,
            }

    narrative = generate_day_narrative(day_date, day_photos, model)
    with get_conn() as conn:
        upsert_narrative(conn, album_id, day_date, narrative, model, photo_ids_json)

    return {
        "day_date": day_date,
        "narrative": narrative,
        "photo_ids": photo_ids,
        "photo_count": len(day_photos),
        "model_used": model,
        "cached": False,
    }


def generate_album_story(
    album_id: int,
    start_date: str | None = None,
    end_date: str | None = None,
    model: str = STORY_LLM_MODEL,
    progress_callback=None,
    force_all: bool = False,
) -> list[dict]:
    """
    Full story generation pipeline for an album.
    Groups photos by day, checks cache, generates missing narratives.
    Returns list of {day_date, narrative, photo_ids, photo_count} dicts.

    progress_callback(current_day_index, total_days, day_date) is called
    before processing each day.
    """
    photos = fetch_album_photos(album_id)
    day_groups = group_photos_by_day(photos, start_date, end_date)

    if not day_groups:
        return []

    total_days = len(day_groups)
    results = []

    for idx, (day_date, day_photos) in enumerate(day_groups.items()):
        if progress_callback:
            progress_callback(idx, total_days, day_date)

        photo_ids = [p["id"] for p in day_photos]
        photo_ids_json = json.dumps(photo_ids)

        # Check cache first
        with get_conn() as conn:
            cached = get_cached_narrative(conn, album_id, day_date)

        # If cache exists, uses the same model, and photo_ids haven't changed (unless forced)
        if not force_all and cached and cached.get("model_used") == model and cached.get("photo_ids") == photo_ids_json:
            narrative = cached["narrative"]
        else:
            # Generate fresh narrative via LLM
            narrative = generate_day_narrative(day_date, day_photos, model)
            # Cache it
            with get_conn() as conn:
                upsert_narrative(conn, album_id, day_date, narrative, model, photo_ids_json)

        results.append({
            "day_date": day_date,
            "narrative": narrative,
            "photo_ids": photo_ids,
            "photo_count": len(day_photos),
        })

    return results
