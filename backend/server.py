"""
FastAPI server: search API + thumbnail serving + static frontend.

Run:
    python -m backend.server
    # or: uvicorn backend.server:app --host 0.0.0.0 --port 8642
"""

import sys
from typing import Optional
import os
import re
import string
import asyncio
import subprocess
import threading
from pathlib import Path
from pydantic import BaseModel

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from PIL import Image, ImageOps
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from backend.config import THUMB_DIR, PREVIEW_DIR, PREVIEW_SIZE, HOST, PORT, SEARCH_TOP_K, SUPPORTED_EXTENSIONS, DATA_DIR
from backend.db import (
    init_db, get_conn, get_image_by_id, get_stats,
    create_album, get_all_albums, get_album_by_id, rename_album,
    add_photo_to_album, remove_photo_from_album,
    get_album_image_ids, get_photo_albums, delete_album,
    bulk_add_photos_to_album, bulk_remove_photos_from_album, bulk_move_photos_to_album,
    get_folder_for_album, get_albums_for_folder,
    delete_images, bulk_update_location,
    get_all_image_paths, update_description, mark_embedded,
    get_faces_for_image, get_face_by_id, insert_face, update_face_person,
    delete_face, get_all_people, get_person_by_id, get_person_by_name,
    upsert_person, update_person, delete_person, get_photos_for_person,
    get_faces_for_person, unlink_face_from_person,
    get_known_face_embeddings, get_untagged_faces_with_embeddings,
    calculate_box_iou,
)
from backend.faces import (
    detect_and_embed_faces, match_face_embedding, crop_face_thumbnail,
    FACES_THUMB_DIR, ensure_models,
)
from backend.search import SemanticSearch
from backend.ingest import tracker, queue_manager, start_background_import
from backend.metadata import search_locations, regenerate_all_thumbnails, generate_thumbnail
from backend.describer import enrich_description


class ImportRequest(BaseModel):
    folder_path: str
    skip_describe: bool = False
    vlm_model: Optional[str] = None
    rescan_mode: str = "incremental"  # "incremental" | "full"
    remove_deleted: bool = True
    target_album_id: Optional[int] = None
    auto_album_sync: bool = True


class SetModelRequest(BaseModel):
    model: str


class CreateAlbumRequest(BaseModel):
    name: str
    description: Optional[str] = ""


class UpdateAlbumRequest(BaseModel):
    name: str
    description: Optional[str] = ""


class AddPhotoRequest(BaseModel):
    image_id: int


class BulkAlbumRequest(BaseModel):
    album_id: int
    image_ids: list[int]


class BulkRemoveAlbumRequest(BaseModel):
    album_id: int
    image_ids: list[int]


class BulkMoveAlbumRequest(BaseModel):
    source_album_id: int
    target_album_id: int
    image_ids: list[int]


class BulkDeleteRequest(BaseModel):
    image_ids: list[int]


class BulkLocationRequest(BaseModel):
    image_ids: list[int]
    place_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: str = ""
    region: str = ""
    country: str = ""


class BulkReprocessRequest(BaseModel):
    image_ids: list[int]
    vlm_model: Optional[str] = None


class UpdateDescriptionRequest(BaseModel):
    description: str


class SaveStoryNoteRequest(BaseModel):
    day_date: str
    title: str
    content: str


class CreateNoteRequest(BaseModel):
    day_date: str = "General"
    title: str
    content: str


class UpdateNoteRequest(BaseModel):
    title: str
    content: str
    day_date: Optional[str] = None


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PixelMemory", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
        "http://localhost:8642",
        "http://127.0.0.1:8642",
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton
_search = None


def get_search() -> SemanticSearch:
    global _search
    if _search is None:
        _search = SemanticSearch()
    return _search


def extract_tags(enriched: str, place: str, camera: str, date_taken: str) -> list[str]:
    """Extract human-readable quick tags from metadata."""
    tags = []
    if place:
        parts = [p.strip() for p in place.split(",") if p.strip()]
        for p in parts[:2]:
            clean = re.sub(r'[^a-zA-Z0-9]', '', p)
            if clean and len(clean) > 2:
                tags.append(f"#{clean}")
    if camera:
        cam_clean = re.sub(r'[^a-zA-Z0-9]', '', camera)
        if cam_clean:
            tags.append(f"#{cam_clean}")
    if date_taken:
        year = date_taken[:4]
        if year.isdigit():
            tags.append(f"#{year}")
    if not tags:
        tags = ["#Photo"]
    return tags[:4]


# ── API routes ───────────────────────────────────────────

@app.get("/api/photos")
def list_all_photos(
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    date_unknown: bool = Query(False),
    geo_only: bool = Query(False),
):
    """Return indexed photos in the library ordered chronologically, with optional date and GPS filters."""
    where_clauses = []
    params = []

    if date_unknown:
        where_clauses.append("(date_taken IS NULL OR date_taken = '')")
    else:
        if start_date and start_date.strip():
            where_clauses.append("date_taken >= ?")
            params.append(start_date.strip())
        if end_date and end_date.strip():
            end_val = end_date.strip()
            if len(end_val) == 10:
                end_val += "T23:59:59"
            where_clauses.append("date_taken <= ?")
            params.append(end_val)
        if (start_date and start_date.strip()) or (end_date and end_date.strip()):
            where_clauses.append("(date_taken IS NOT NULL AND date_taken != '')")

    if geo_only:
        where_clauses.append("(latitude IS NOT NULL AND longitude IS NOT NULL)")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT * FROM images 
            {where_sql}
            ORDER BY COALESCE(date_taken, '') DESC, id DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset]).fetchall()

        total_row = conn.execute(f"SELECT COUNT(*) as total FROM images {where_sql}", params).fetchone()
        total = total_row["total"] if total_row else 0

        results = []
        for row in rows:
            enriched = row["enriched_text"] or ""
            place = row["place_name"] or ""
            camera = row["camera_model"] or ""
            dt = row["date_taken"] or ""
            results.append({
                "id": row["id"],
                "file_path": row["file_path"],
                "score": 1.0,
                "raw_score": 1.0,
                "enriched_text": enriched,
                "raw_description": row["raw_description"] or "",
                "date_taken": row["date_taken"],
                "place_name": place,
                "camera_model": camera,
                "camera_make": row["camera_make"] or "",
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "tags": extract_tags(enriched, place, camera, dt),
            })

    return {
        "count": len(results),
        "total": total,
        "results": results,
    }


@app.get("/api/search")
def search_images(
    q: str = Query("", min_length=0),
    top_k: int = Query(SEARCH_TOP_K, ge=1, le=200),
    filter_mode: str = Query("balanced"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    date_unknown: bool = Query(False),
    geo_only: bool = Query(False),
):
    """Semantic search over image descriptions with hybrid scoring, relevance filtering, and date/geo criteria."""
    q_str = q.strip() if q else ""
    if not q_str:
        return list_all_photos(
            limit=top_k,
            start_date=start_date,
            end_date=end_date,
            date_unknown=date_unknown,
            geo_only=geo_only,
        )

    engine = get_search()
    # If filters are active, retrieve more candidates from vector space to filter down
    fetch_k = top_k * 3 if (start_date or end_date or date_unknown or geo_only) else top_k
    hits = engine.query(q_str, top_k=fetch_k, filter_mode=filter_mode)

    results = []
    with get_conn() as conn:
        for hit in hits:
            row = get_image_by_id(conn, hit["image_id"])
            if row is None:
                continue

            dt = row["date_taken"] or ""
            if date_unknown:
                if dt:
                    continue
            else:
                if start_date and start_date.strip():
                    if not dt or dt < start_date.strip():
                        continue
                if end_date and end_date.strip():
                    end_val = end_date.strip()
                    if len(end_val) == 10:
                        end_val += "T23:59:59"
                    if not dt or dt > end_val:
                        continue

            if geo_only:
                if not row["latitude"] or not row["longitude"]:
                    continue

            enriched = hit["enriched_text"] or ""
            place = row["place_name"] or ""
            camera = row["camera_model"] or ""
            tags = extract_tags(enriched, place, camera, dt)
            results.append({
                "id": row["id"],
                "file_path": row["file_path"],
                "score": round(hit["score"], 4),
                "raw_score": hit.get("raw_score", round(hit["score"], 4)),
                "enriched_text": enriched,
                "raw_description": row["raw_description"] or "",
                "date_taken": row["date_taken"],
                "place_name": place,
                "camera_model": camera,
                "camera_make": row["camera_make"] or "",
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "tags": tags,
            })
            if len(results) >= top_k:
                break

    return {
        "query": q_str,
        "filter_mode": filter_mode,
        "count": len(results),
        "total_in_db": engine.count,
        "results": results,
    }


def _format_photo_dict(row, score: float = 1.0, raw_score: float = 1.0, enriched: str = "") -> dict:
    place = row["place_name"] or ""
    camera = row["camera_model"] or ""
    dt = row["date_taken"] or ""
    enr = enriched or row["enriched_text"] or row["raw_description"] or ""
    tags = extract_tags(enr, place, camera, dt)
    return {
        "id": row["id"],
        "file_path": row["file_path"],
        "score": round(score, 4),
        "raw_score": round(raw_score, 4),
        "enriched_text": enr,
        "raw_description": row["raw_description"] or "",
        "date_taken": row["date_taken"],
        "place_name": place,
        "camera_model": camera,
        "camera_make": row["camera_make"] or "",
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "tags": tags,
    }


def parse_ai_query_intent(prompt: str, known_people_names: list[str]) -> dict:
    """Fast, deterministic NLP heuristic query parser for search intent extraction."""
    import re

    cleaned = prompt.strip()
    lower = cleaned.lower()

    # 1. Detect Person / Pet mentions
    matched_people = []
    for name in known_people_names:
        if re.search(r'\b' + re.escape(name.lower()) + r'\b', lower):
            matched_people.append(name)

    # 2. Detect Date filters
    date_mode = "all"
    start_date = None
    end_date = None
    date_unknown = False

    if any(k in lower for k in ("unknown date", "no date", "without date", "missing date", "undated")):
        date_unknown = True
        date_mode = "unknown"
    elif any(k in lower for k in ("past 7 days", "last 7 days", "past week", "last week")):
        date_mode = "past_7d"
    elif any(k in lower for k in ("past 30 days", "last 30 days", "past month", "last month")):
        date_mode = "past_30d"
    else:
        # Check specific 4-digit years (e.g. 2018-2030)
        year_match = re.search(r'\b(201\d|202\d|203\d)\b', lower)
        if year_match:
            year = year_match.group(1)
            months = {
                "january": "01", "jan": "01",
                "february": "02", "feb": "02",
                "march": "03", "mar": "03",
                "april": "04", "apr": "04",
                "may": "05",
                "june": "06", "jun": "06",
                "july": "07", "jul": "07",
                "august": "08", "aug": "08",
                "september": "09", "sep": "09", "sept": "09",
                "october": "10", "oct": "10",
                "november": "11", "nov": "11",
                "december": "12", "dec": "12",
            }
            found_month = None
            for m_name, m_num in months.items():
                if re.search(r'\b' + m_name + r'\b', lower):
                    found_month = m_num
                    break
            if found_month:
                start_date = f"{year}-{found_month}-01"
                end_date = f"{year}-{found_month}-31"
                date_mode = "custom"
            else:
                date_mode = year
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"

    # 3. Detect Geo / Location / GPS constraints
    geo_only = False
    if any(k in lower for k in ("gps", "coordinates", "geotagged", "with location", "has location")):
        geo_only = True

    # 4. Detect Group By
    group_by = "none"
    if "group by day" in lower or "grouped by day" in lower:
        group_by = "day"
    elif "group by month" in lower or "grouped by month" in lower:
        group_by = "month"
    elif "group by year" in lower or "grouped by year" in lower:
        group_by = "year"
    elif "group by location" in lower or "group by place" in lower:
        group_by = "location"
    elif "group by camera" in lower:
        group_by = "camera"

    # 5. Detect Sort Order
    sort = "relevance"
    if any(k in lower for k in ("newest", "most recent", "latest")):
        sort = "newest"
    elif any(k in lower for k in ("oldest", "earliest")):
        sort = "oldest"

    # 6. Extract Clean Semantic Core Query
    core = re.sub(r'^(show me|find|search for|look for|get|display|list|can you find|please find)\s+(all\s+)?(my\s+)?(photos|pictures|images|memories|shots)?(\s+of|\s+with|\s+from)?\s*', '', lower, flags=re.IGNORECASE)
    core = re.sub(r'\b(group\s+by\s+\w+|grouped\s+by\s+\w+)\b', '', core, flags=re.IGNORECASE)
    core = re.sub(r'\b(in\s+(201\d|202\d|203\d)|taken\s+in\s+\w+\s+\d{4}|from\s+(201\d|202\d|203\d))\b', '', core, flags=re.IGNORECASE)
    core = re.sub(r'\b(with\s+gps|with\s+location|geotagged|without\s+date|unknown\s+date|undated)\b', '', core, flags=re.IGNORECASE)
    core = re.sub(r'\b(newest\s+first|oldest\s+first|most\s+recent)\b', '', core, flags=re.IGNORECASE)
    for name in matched_people:
        core = re.sub(r'\b(with|of|and)?\s*' + re.escape(name.lower()) + r'\b', '', core, flags=re.IGNORECASE)

    core = re.sub(r'\s+', ' ', core).strip()
    if not core and not matched_people and not date_unknown and not start_date and not geo_only:
        core = cleaned

    return {
        "semantic_query": core,
        "matched_people": matched_people,
        "date_mode": date_mode,
        "start_date": start_date,
        "end_date": end_date,
        "date_unknown": date_unknown,
        "geo_only": geo_only,
        "group_by": group_by,
        "sort": sort,
    }


def query_fast_llm_intent(prompt: str, known_people: list[str]) -> Optional[dict]:
    """
    Attempt ultra-fast structured extraction using non-reasoning local LLM (e.g., gemma4:e2b) via Ollama.
    Uses num_predict: 60, temperature: 0.0 and tight timeout (<750ms) to ensure zero UI freezing.
    """
    import urllib.request
    import json
    try:
        req_payload = {
            "model": "gemma4:e2b",
            "prompt": f"""Extract photo search filters as JSON: {{"semantic_query": "visual keywords", "people": [], "year": null, "geo_only": false, "group_by": null, "sort": null}}.
Known people: {json.dumps(known_people)}
Query: "{prompt}"
JSON:""",
            "stream": False,
            "options": {
                "num_predict": 60,
                "temperature": 0.0,
                "top_p": 0.9,
            },
            "format": "json"
        }
        req_data = json.dumps(req_payload).encode('utf-8')
        http_req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(http_req, timeout=0.75) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode('utf-8'))
                raw_json = body.get("response", "").strip()
                parsed = json.loads(raw_json)
                return parsed
    except Exception:
        pass
    return None


class AIAskRequest(BaseModel):
    prompt: str
    top_k: Optional[int] = 60
    filter_mode: Optional[str] = "balanced"


@app.post("/api/ai/ask")
def ai_ask_search(req: AIAskRequest):
    """
    Intelligent AI query understanding and multi-dimensional semantic execution.
    Extracts semantic visual concepts, person/pet tags, date ranges, GPS geolocation, and sorting.
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        all_p = list_all_photos(limit=req.top_k)
        return {
            "prompt": "",
            "semantic_query": "",
            "explanation": "Showing all photos in library",
            "filters": {
                "date_mode": "all",
                "start_date": None,
                "end_date": None,
                "date_unknown": False,
                "geo_only": False,
                "group_by": "none",
                "sort": "relevance",
                "matched_people": [],
            },
            "results": all_p.get("results", []),
            "count": all_p.get("count", 0),
            "suggestions": ["Find photos from 2026", "Photos with GPS", "Show pets"]
        }

    known_people = []
    people_id_map = {}
    with get_conn() as conn:
        p_rows = conn.execute("SELECT id, name FROM people").fetchall()
        for r in p_rows:
            name = r["name"]
            known_people.append(name)
            people_id_map[name.lower()] = r["id"]

    plan = parse_ai_query_intent(prompt, known_people)
    semantic_q = plan["semantic_query"]
    date_mode = plan["date_mode"]
    start_date = plan["start_date"]
    end_date = plan["end_date"]
    date_unknown = plan["date_unknown"]
    geo_only = plan["geo_only"]
    matched_people = plan["matched_people"]
    group_by = plan["group_by"]
    sort = plan["sort"]

    engine = get_search()
    results = []

    person_image_ids = set()
    if matched_people:
        with get_conn() as conn:
            for p_name in matched_people:
                pid = people_id_map.get(p_name.lower())
                if pid:
                    p_imgs = get_photos_for_person(conn, pid)
                    for img in p_imgs:
                        person_image_ids.add(img["id"])

    if not semantic_q and person_image_ids:
        with get_conn() as conn:
            for img_id in person_image_ids:
                row = get_image_by_id(conn, img_id)
                if not row:
                    continue
                dt = row["date_taken"] or ""
                if date_unknown and dt:
                    continue
                if start_date and (not dt or dt < start_date):
                    continue
                if end_date and (not dt or dt > end_date):
                    continue
                if geo_only and (not row["latitude"] or not row["longitude"]):
                    continue
                results.append(_format_photo_dict(row, 0.98))
    elif not semantic_q and (date_unknown or start_date or geo_only):
        all_p = list_all_photos(
            limit=req.top_k,
            start_date=start_date,
            end_date=end_date,
            date_unknown=date_unknown,
            geo_only=geo_only,
        )
        results = all_p.get("results", [])
    else:
        fetch_k = req.top_k * 3 if (start_date or end_date or date_unknown or geo_only or person_image_ids) else req.top_k
        hits = engine.query(semantic_q if semantic_q else prompt, top_k=fetch_k, filter_mode=req.filter_mode)

        with get_conn() as conn:
            for hit in hits:
                row = get_image_by_id(conn, hit["image_id"])
                if row is None:
                    continue

                dt = row["date_taken"] or ""
                if date_unknown:
                    if dt:
                        continue
                else:
                    if start_date and start_date.strip():
                        if not dt or dt < start_date.strip():
                            continue
                    if end_date and end_date.strip():
                        end_val = end_date.strip()
                        if len(end_val) == 10:
                            end_val += "T23:59:59"
                        if not dt or dt > end_val:
                            continue

                if geo_only:
                    if not row["latitude"] or not row["longitude"]:
                        continue

                score = hit["score"]
                if person_image_ids:
                    if row["id"] in person_image_ids:
                        score = min(0.99, score + 0.15)
                    elif len(matched_people) > 0 and len(semantic_q) < 3:
                        continue

                results.append(_format_photo_dict(row, score, hit.get("raw_score", score), hit.get("enriched_text", "")))
                if len(results) >= req.top_k:
                    break

    if sort == "newest":
        results.sort(key=lambda x: x.get("date_taken") or "", reverse=True)
    elif sort == "oldest":
        results.sort(key=lambda x: x.get("date_taken") or "9999-99-99")

    # Construct AI explanation
    explanation_parts = []
    if semantic_q:
        explanation_parts.append(f"Searching visual scenes for **'{semantic_q}'**")
    if matched_people:
        people_str = ", ".join(f"**{p}**" for p in matched_people)
        explanation_parts.append(f"tagged with {people_str}")
    if date_mode == "unknown":
        explanation_parts.append("filtered to **date unknown**")
    elif start_date and end_date:
        if start_date[:4] == end_date[:4] and start_date[5:7] == end_date[5:7]:
            explanation_parts.append(f"taken in **{start_date[:7]}**")
        elif start_date[:4] == end_date[:4]:
            explanation_parts.append(f"taken in **{start_date[:4]}**")
        else:
            explanation_parts.append(f"from **{start_date}** to **{end_date}**")
    elif date_mode in ("past_7d", "past_30d"):
        explanation_parts.append(f"taken in the **{date_mode.replace('_', ' ')}**")
    if geo_only:
        explanation_parts.append("with **GPS coordinates**")
    if group_by != "none":
        explanation_parts.append(f"grouped by **{group_by}**")

    explanation = " · ".join(explanation_parts) if explanation_parts else f"Searching for '{prompt}'"

    suggestions = []
    if not geo_only:
        suggestions.append(f"{prompt} with GPS")
    if group_by == "none":
        suggestions.append(f"{prompt} grouped by month")
    if date_mode == "all":
        suggestions.append(f"{prompt} from 2026")

    return {
        "prompt": prompt,
        "semantic_query": semantic_q,
        "explanation": explanation,
        "filters": {
            "date_mode": date_mode,
            "start_date": start_date,
            "end_date": end_date,
            "date_unknown": date_unknown,
            "geo_only": geo_only,
            "group_by": group_by,
            "sort": sort,
            "matched_people": matched_people,
        },
        "count": len(results),
        "total_in_db": engine.count,
        "suggestions": suggestions[:3],
        "results": results,
    }


@app.get("/api/tags")
def get_dynamic_tags():
    """Return common dynamic tags across all indexed photos."""
    with get_conn() as conn:
        rows = conn.execute("SELECT place_name, camera_model, date_taken FROM images LIMIT 500").fetchall()

    tag_counts = {}
    for r in rows:
        for t in extract_tags("", r["place_name"] or "", r["camera_model"] or "", r["date_taken"] or ""):
            tag_counts[t] = tag_counts.get(t, 0) + 1

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    return {"tags": [t[0] for t in sorted_tags[:16]]}


@app.get("/api/system")
def get_system_status():
    """Return local AI model and hardware status."""
    hw_name = "CPU (PyTorch)"
    vram = 0.0
    cuda_avail = False
    try:
        import torch
        if torch.cuda.is_available():
            cuda_avail = True
            hw_name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
    except Exception:
        pass

    from backend.describer import is_ollama_ready, get_active_vlm_model, get_available_vlm_models
    from backend.config import OLLAMA_HOST

    active_model = get_active_vlm_model()
    ollama_ok = is_ollama_ready(OLLAMA_HOST, active_model)
    vlm_label = f"Ollama · {active_model}" if ollama_ok else "Moondream2 (PyTorch)"

    with get_conn() as conn:
        stats = get_stats(conn)

    search = get_search()
    return {
        "cuda_available": cuda_avail,
        "device_name": hw_name,
        "vram_gb": vram,
        "vlm_provider": "Ollama" if ollama_ok else "PyTorch",
        "vlm_model": vlm_label,
        "active_vlm_model": active_model,
        "available_vlm_models": get_available_vlm_models(),
        "ollama_ready": ollama_ok,
        "ollama_host": OLLAMA_HOST,
        "embedding_model": "all-MiniLM-L6-v2",
        "total_images": stats.get("total", 0),
        "total_vectors": search.count,
    }


@app.get("/api/models/vlm")
def get_vlm_models_endpoint():
    """Return available VLM models and current active selection."""
    from backend.describer import get_available_vlm_models, get_active_vlm_model
    return {
        "active_model": get_active_vlm_model(),
        "models": get_available_vlm_models(),
    }


@app.post("/api/models/vlm")
def set_vlm_model_endpoint(req: SetModelRequest):
    """Set global active VLM model."""
    from backend.describer import set_active_vlm_model, get_available_vlm_models
    active = set_active_vlm_model(req.model)
    return {
        "status": "ok",
        "active_model": active,
        "models": get_available_vlm_models(),
    }


@app.get("/api/stats")
def pipeline_stats():
    """Current ingest pipeline statistics."""
    with get_conn() as conn:
        stats = get_stats(conn)
    search = get_search()
    stats["vectors"] = search.count
    return stats


@app.get("/api/image/{image_id}")
def get_image_info(image_id: int):
    """Full metadata for a single image."""
    with get_conn() as conn:
        row = get_image_by_id(conn, image_id)
    if row is None:
        raise HTTPException(404, "Image not found")
    return dict(row)


@app.post("/api/demo/seed")
def seed_demo_endpoint():
    """Seed sample photos and index for instant exploration."""
    from backend.demo import seed_demo_archive
    res = seed_demo_archive(clear_existing=False)
    return {
        "status": "ok",
        "message": f"Seeded {res['seeded']} sample memories",
        "total_vectors": res["total_vectors"],
    }


@app.post("/api/database/reset")
def reset_database_endpoint():
    """Clear all images from the database and vector store."""
    with get_conn() as conn:
        conn.execute("DELETE FROM images")
        conn.execute("DELETE FROM albums")
        conn.execute("DELETE FROM album_images")
    search = get_search()
    search.reset()
    return {"status": "ok", "message": "Database and vector index cleared"}


# ── Import Pipeline Routes ───────────────────────────────

@app.get("/api/folders")
def list_indexed_folders_endpoint():
    """List all unique source directories currently indexed in the library with photo counts."""
    with get_conn() as conn:
        rows = conn.execute("SELECT file_path FROM images").fetchall()

    folders_map = {}
    for r in rows:
        fp = Path(r["file_path"])
        parent = str(fp.parent)
        if parent not in folders_map:
            folders_map[parent] = {
                "folder_path": parent,
                "folder_name": fp.parent.name or parent,
                "count": 0,
            }
        folders_map[parent]["count"] += 1

    return {"folders": sorted(list(folders_map.values()), key=lambda x: x["count"], reverse=True)}


@app.post("/api/import/start")
def start_import_endpoint(req: ImportRequest):
    """Start directory scan and AI ingestion or queue it if already running."""
    fpath = Path(req.folder_path).resolve()
    if not fpath.exists() or not fpath.is_dir():
        raise HTTPException(400, f"Invalid folder directory: '{req.folder_path}' does not exist on disk.")

    res = queue_manager.enqueue(
        str(fpath),
        skip_describe=req.skip_describe,
        vlm_model=req.vlm_model,
        rescan_mode=req.rescan_mode,
        remove_deleted=req.remove_deleted,
        target_album_id=req.target_album_id,
        auto_album_sync=req.auto_album_sync,
    )
    if res["status"] == "started":
        return {
            "status": "started",
            "message": f"Started import of {fpath.name}",
            "queue_position": 1,
            "queue_length": 1,
            "tracker": tracker.to_dict(),
        }
    elif res["status"] == "queued":
        return {
            "status": "queued",
            "message": f"Added '{fpath.name}' to import queue (position #{res['queue_position']})",
            "queue_position": res["queue_position"],
            "queue_length": res["queue_length"],
            "tracker": tracker.to_dict(),
        }
    else:
        return {
            "status": "already_queued",
            "message": f"'{fpath.name}' is already in the import queue (position #{res['queue_position']}).",
            "queue_position": res["queue_position"],
            "queue_length": res["queue_length"],
            "tracker": tracker.to_dict(),
        }


@app.post("/api/import/rescan")
def rescan_folder_endpoint(req: ImportRequest):
    """
    Rescan an existing folder.
    In 'incremental' mode: skips existing photos, adds new, prunes deleted.
    In 'full' mode: re-analyzes all photos in the folder from scratch, prunes deleted.
    """
    raw_path = (req.folder_path or "").strip().strip("\"'")
    fpath = Path(raw_path).expanduser()
    if not fpath.is_absolute():
        fpath = (Path.cwd() / fpath).resolve()
    else:
        fpath = fpath.resolve()
    if not fpath.exists() or not fpath.is_dir():
        raise HTTPException(400, f"Invalid folder directory: '{req.folder_path}' does not exist on disk.")

    res = queue_manager.enqueue(
        str(fpath),
        skip_describe=req.skip_describe,
        vlm_model=req.vlm_model,
        rescan_mode=req.rescan_mode,
        remove_deleted=req.remove_deleted,
        target_album_id=req.target_album_id,
        auto_album_sync=req.auto_album_sync,
    )
    res["is_rescan"] = True
    res["folder_path"] = str(fpath).replace("\\", "/")
    mode_name = "Incremental Sync" if req.rescan_mode == "incremental" else "Full Rescan"
    if res["status"] == "started":
        res["message"] = f"Started {mode_name} for '{fpath.name}'..."
    elif res["status"] == "queued":
        res["message"] = f"Added '{fpath.name}' ({mode_name}) to queue (position #{res['queue_position']})"
    return res


def _open_folder_dialog_sync() -> Optional[str]:
    # 1. macOS: Native Finder Folder Picker via AppleScript / osascript (instant & native on macOS)
    if sys.platform == "darwin":
        try:
            ascript = (
                'tell application "System Events"\n'
                '    activate\n'
                '    set folderChosen to choose folder with prompt "Select Photo Folder to Import or Rescan"\n'
                '    POSIX path of folderChosen\n'
                'end tell'
            )
            proc = subprocess.run(
                ["osascript", "-e", ascript],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0:
                out = proc.stdout.strip()
                if out:
                    p = Path(out).expanduser().resolve()
                    if p.exists() and p.is_dir():
                        return str(p).replace("\\", "/")
            else:
                # Fallback simple osascript without System Events wrapper
                proc2 = subprocess.run(
                    ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select Photo Folder to Import or Rescan")'],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if proc2.returncode == 0:
                    out2 = proc2.stdout.strip()
                    if out2:
                        p = Path(out2).expanduser().resolve()
                        if p.exists() and p.is_dir():
                            return str(p).replace("\\", "/")
        except Exception as e:
            print(f"macOS osascript folder picker error: {e}")

    # 2. Linux: Zenity or Kdialog
    if sys.platform.startswith("linux"):
        try:
            proc = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title=Select Photo Folder to Import or Rescan"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0:
                out = proc.stdout.strip()
                if out:
                    p = Path(out).expanduser().resolve()
                    if p.exists() and p.is_dir():
                        return str(p).replace("\\", "/")
        except Exception:
            pass
        try:
            proc = subprocess.run(
                ["kdialog", "--getexistingdirectory", "--title", "Select Photo Folder to Import or Rescan"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0:
                out = proc.stdout.strip()
                if out:
                    p = Path(out).expanduser().resolve()
                    if p.exists() and p.is_dir():
                        return str(p).replace("\\", "/")
        except Exception:
            pass

    # 3. Windows: PowerShell FolderBrowserDialog with TopMost Form
    if sys.platform.startswith("win"):
        ps_cmd = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dlg.Description = 'Select Photo Folder to Import or Rescan'; "
            "$dlg.ShowNewFolderButton = $false; "
            "$form = New-Object System.Windows.Forms.Form; "
            "$form.TopMost = $true; "
            "if ($dlg.ShowDialog($form) -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output ('PICKED:' + $dlg.SelectedPath) }"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=120,
            )
            for line in proc.stdout.splitlines():
                if line.startswith("PICKED:"):
                    p = line[len("PICKED:"):].strip()
                    if p and Path(p).exists() and Path(p).is_dir():
                        return str(Path(p).resolve()).replace("\\", "/")
        except Exception as e:
            print(f"PowerShell picker error: {e}")

    # 4. Universal Fallback: Tkinter in an isolated subprocess
    py_code = (
        "import sys\n"
        "try:\n"
        "    import tkinter as tk\n"
        "    from tkinter import filedialog\n"
        "    root = tk.Tk()\n"
        "    root.withdraw()\n"
        "    try:\n"
        "        root.wm_attributes('-topmost', 1)\n"
        "    except Exception:\n"
        "        pass\n"
        "    path = filedialog.askdirectory(parent=root, title='Select Photo Folder to Import or Rescan')\n"
        "    root.destroy()\n"
        "    if path:\n"
        "        print('PICKED:' + path)\n"
        "except Exception as e:\n"
        "    sys.stderr.write(str(e))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", py_code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("PICKED:"):
                p = line[len("PICKED:"):].strip()
                if p and Path(p).exists() and Path(p).is_dir():
                    return str(Path(p).resolve()).replace("\\", "/")
    except Exception as e:
        print(f"Tkinter picker subprocess error: {e}")

    return None


@app.get("/api/system/browse-folder")
async def browse_folder_endpoint():
    """Open native OS Folder Browser dialog and return selected path."""
    selected = await asyncio.to_thread(_open_folder_dialog_sync)
    if not selected:
        return {"status": "cancelled", "path": None}
    return {"status": "ok", "path": selected}


@app.get("/api/system/validate-folder")
def validate_folder_endpoint(path: str = ""):
    """Validate if a folder path exists and is a directory on disk, and count images."""
    if not path or not path.strip():
        return {
            "valid": False,
            "exists": False,
            "is_dir": False,
            "resolved_path": "",
            "photo_count": 0,
            "message": "Path is empty",
        }

    clean_input = path.strip().strip("\"'")
    try:
        p = Path(clean_input).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()

        if not p.exists():
            return {
                "valid": False,
                "exists": False,
                "is_dir": False,
                "resolved_path": str(p).replace("\\", "/"),
                "photo_count": 0,
                "message": "Folder does not exist on disk",
            }

        if not p.is_dir():
            return {
                "valid": False,
                "exists": True,
                "is_dir": False,
                "resolved_path": str(p).replace("\\", "/"),
                "photo_count": 0,
                "message": "Path exists but is a file, not a directory",
            }

        # Fast photo count
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tiff", ".tif", ".bmp", ".avif", ".raw", ".cr2", ".nef", ".arw", ".dng"}
        count = 0
        try:
            for root_dir, _, files in os.walk(p):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in valid_exts and not f.startswith("."):
                        count += 1
                        if count >= 10000:
                            break
                if count >= 10000:
                    break
        except Exception:
            pass

        return {
            "valid": True,
            "exists": True,
            "is_dir": True,
            "resolved_path": str(p).replace("\\", "/"),
            "photo_count": count,
            "message": f"Valid directory ({count} photo{'s' if count != 1 else ''} found)",
        }
    except Exception as e:
        return {
            "valid": False,
            "exists": False,
            "is_dir": False,
            "resolved_path": "",
            "photo_count": 0,
            "message": f"Invalid path syntax: {str(e)}",
        }


@app.get("/api/system/folders")
def list_system_folders_endpoint(query: Optional[str] = None, parent: Optional[str] = None):
    """
    Search or browse folders on the local computer.
    Returns quick locations (Drives, User folders, Project galleries) and matching or child directories.
    """
    quick_locations = []

    # 1. Available Drives (Windows)
    if sys.platform.startswith("win"):
        for d in string.ascii_uppercase:
            drive_path = f"{d}:/"
            if os.path.exists(f"{d}:"):
                quick_locations.append({"name": f"Drive ({d}:)", "path": drive_path, "type": "drive"})
    elif sys.platform == "darwin":
        if os.path.exists("/Volumes"):
            try:
                for v in os.listdir("/Volumes"):
                    vpath = Path("/Volumes") / v
                    if vpath.is_dir() and not v.startswith("."):
                        quick_locations.append({"name": f"Volume ({v})", "path": str(vpath).replace("\\", "/"), "type": "drive"})
            except Exception:
                pass
    elif sys.platform.startswith("linux"):
        if os.path.exists("/media"):
            try:
                for user_media in os.listdir("/media"):
                    um_path = Path("/media") / user_media
                    if um_path.is_dir():
                        for v in os.listdir(um_path):
                            vpath = um_path / v
                            if vpath.is_dir():
                                quick_locations.append({"name": f"Drive ({v})", "path": str(vpath).replace("\\", "/"), "type": "drive"})
            except Exception:
                pass

    # 2. Common User folders & Project Gallery
    home = Path.home()
    candidates = [
        ("Pictures", home / "OneDrive" / "Pictures" if (home / "OneDrive" / "Pictures").exists() else home / "Pictures"),
        ("Desktop", home / "Desktop"),
        ("Downloads", home / "Downloads"),
        ("Documents", home / "Documents"),
        ("Home", home),
        ("Project Gallery", Path(DATA_DIR).resolve()),
    ]
    for name, p in candidates:
        if p.exists() and p.is_dir():
            clean_p = str(p.resolve()).replace("\\", "/")
            if not any(q["path"].lower() == clean_p.lower() for q in quick_locations):
                quick_locations.append({"name": name, "path": clean_p, "type": "folder"})

    # 3. Explore parent or search
    results = []
    target_dir = None
    if parent:
        p_obj = Path(parent).expanduser().resolve()
        if p_obj.exists() and p_obj.is_dir():
            target_dir = p_obj

    if target_dir:
        try:
            with os.scandir(target_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                        clean_sub = str(Path(entry.path).resolve()).replace("\\", "/")
                        results.append({
                            "name": entry.name,
                            "path": clean_sub,
                            "parent": str(target_dir.resolve()).replace("\\", "/"),
                        })
        except Exception:
            pass
        results.sort(key=lambda x: x["name"].lower())

    elif query and len(query.strip()) >= 2:
        q_lower = query.strip().lower()
        # Only search sensible user folders, never entire drives from root C:/
        search_roots = [Path(q["path"]) for q in quick_locations if q.get("type") == "folder" and Path(q["path"]).exists()]
        project_gallery = (Path.cwd() / "gallery").resolve()
        if project_gallery.exists() and project_gallery not in search_roots:
            search_roots.append(project_gallery)

        # If user typed a path directly (e.g. C:/... or E:/...), search within that folder
        if ("/" in query or "\\" in query or ":" in query):
            try:
                cand_path = Path(query).resolve()
                parent_cand = cand_path if (cand_path.exists() and cand_path.is_dir()) else cand_path.parent
                if parent_cand.exists() and parent_cand.is_dir() and parent_cand not in search_roots:
                    search_roots.insert(0, parent_cand)
            except Exception:
                pass

        seen_paths = set()
        for s_root in search_roots:
            if not s_root.exists():
                continue
            try:
                for root_dir, dirnames, _ in os.walk(s_root):
                    # Limit depth to 3 levels from s_root for fast response
                    rel_depth = len(Path(root_dir).resolve().parts) - len(s_root.resolve().parts)
                    if rel_depth >= 3:
                        dirnames.clear()
                        continue
                    dirnames[:] = [
                        d for d in dirnames
                        if not d.startswith(".")
                        and d.lower() not in ("node_modules", "venv", ".venv", "__pycache__", "windows", "program files", "appdata", "system volume information")
                    ]
                    for d in dirnames:
                        if q_lower in d.lower():
                            full_p = str(Path(os.path.join(root_dir, d)).resolve()).replace("\\", "/")
                            if full_p.lower() not in seen_paths:
                                seen_paths.add(full_p.lower())
                                results.append({
                                    "name": d,
                                    "path": full_p,
                                    "parent": str(Path(root_dir).resolve()).replace("\\", "/"),
                                })
                                if len(results) >= 25:
                                    break
                    if len(results) >= 25:
                        break
            except Exception:
                continue

    return {
        "status": "ok",
        "quick_locations": quick_locations,
        "results": results,
    }


@app.get("/api/import/status")
def get_import_status_endpoint():
    """Poll live progress, queue status, and ETA for the active import."""
    status = tracker.to_dict()
    status.update(queue_manager.get_queue_info())
    return status



@app.post("/api/import/cancel")
def cancel_import_endpoint():
    """Request cancellation of running import and clear remaining queue."""
    queue_manager.cancel()
    return {"status": "cancelled", "tracker": tracker.to_dict()}


# ── Album Routes ─────────────────────────────────────────

@app.get("/api/albums")
def list_albums_endpoint():
    """List all created photo albums."""
    with get_conn() as conn:
        albums = get_all_albums(conn)
    return {"albums": albums}


@app.post("/api/albums")
def create_album_endpoint(req: CreateAlbumRequest):
    """Create a new photo album."""
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Album title cannot be blank.")
    try:
        with get_conn() as conn:
            album_id = create_album(conn, name, req.description)
            album = get_album_by_id(conn, album_id)
            album["photo_count"] = 0
            album["cover_image_id"] = None
            return album
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(400, f"An album named '{name}' already exists.")
        raise HTTPException(500, f"Failed to create album: {e}")


@app.get("/api/albums/{album_id}/folder")
def get_album_folder_endpoint(album_id: int):
    """Detect the folder associated with an album based on its member photos."""
    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")
        detected_folder = get_folder_for_album(conn, album_id)
    return {
        "status": "ok",
        "album_id": album_id,
        "album_name": album["name"],
        "folder_path": detected_folder,
    }


@app.get("/api/albums/{album_id}")
def get_album_endpoint(
    album_id: int,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    date_unknown: bool = Query(False),
    geo_only: bool = Query(False),
):
    """Get album metadata and all photos in this album, with optional date and geo filters."""
    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")

        image_ids = get_album_image_ids(conn, album_id)
        photos = []
        for img_id in image_ids:
            row = get_image_by_id(conn, img_id)
            if not row:
                continue

            dt = row["date_taken"] or ""
            if date_unknown:
                if dt:
                    continue
            else:
                if start_date and start_date.strip():
                    if not dt or dt < start_date.strip():
                        continue
                if end_date and end_date.strip():
                    end_val = end_date.strip()
                    if len(end_val) == 10:
                        end_val += "T23:59:59"
                    if not dt or dt > end_val:
                        continue

            if geo_only:
                if not row["latitude"] or not row["longitude"]:
                    continue

            enriched = row["enriched_text"] or ""
            place = row["place_name"] or ""
            camera = row["camera_model"] or ""
            photos.append({
                "id": row["id"],
                "file_path": row["file_path"],
                "score": 1.0,
                "raw_score": 1.0,
                "enriched_text": enriched,
                "raw_description": row["raw_description"] or "",
                "date_taken": row["date_taken"],
                "place_name": place,
                "camera_model": camera,
                "camera_make": row["camera_make"] or "",
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "tags": extract_tags(enriched, place, camera, dt),
            })

    album["photo_count"] = len(photos)
    album["photos"] = photos
    return album


@app.post("/api/albums/{album_id}/photos")
def add_photo_endpoint(album_id: int, req: AddPhotoRequest):
    """Add a photo to an album."""
    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")
        img = get_image_by_id(conn, req.image_id)
        if not img:
            raise HTTPException(404, f"Photo {req.image_id} not found.")

        add_photo_to_album(conn, album_id, req.image_id)
    return {"status": "ok", "album_id": album_id, "image_id": req.image_id, "album_name": album["name"]}


@app.delete("/api/albums/{album_id}/photos/{image_id}")
def remove_photo_endpoint(album_id: int, image_id: int):
    """Remove a photo from an album."""
    with get_conn() as conn:
        remove_photo_from_album(conn, album_id, image_id)
    return {"status": "ok", "album_id": album_id, "image_id": image_id}


@app.put("/api/albums/{album_id}")
def update_album_endpoint(album_id: int, req: UpdateAlbumRequest):
    """Rename or update an album."""
    new_name = req.name.strip()
    if not new_name:
        raise HTTPException(400, "Album name cannot be empty.")
    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")
        try:
            rename_album(conn, album_id, new_name, req.description or "")
        except Exception as e:
            if "UNIQUE constraint failed" in str(e):
                raise HTTPException(400, f"An album named '{new_name}' already exists.")
            raise HTTPException(500, f"Database error: {e}")

        updated = get_album_by_id(conn, album_id)
        return {"status": "ok", "album": updated}


@app.delete("/api/albums/{album_id}")
def delete_album_endpoint(album_id: int):
    """Delete an album."""
    with get_conn() as conn:
        delete_album(conn, album_id)
    return {"status": "ok", "deleted_id": album_id}



@app.get("/api/photos/{image_id}/albums")
def get_photo_albums_endpoint(image_id: int):
    """Get all albums containing a specific photo."""
    with get_conn() as conn:
        albums = get_photo_albums(conn, image_id)
    return {"albums": albums}


@app.put("/api/photos/{image_id}/description")
def update_photo_description(image_id: int, req: UpdateDescriptionRequest):
    """Update a photo's description manually, re-enrich metadata, and re-embed in Zvec."""
    raw_desc = req.description.strip()
    with get_conn() as conn:
        row = get_image_by_id(conn, image_id)
        if not row:
            raise HTTPException(404, f"Photo {image_id} not found.")

        meta = {
            "date_taken": row["date_taken"],
            "place_name": row["place_name"],
            "camera_model": row["camera_model"],
        }
        enriched = enrich_description(raw_desc, meta)
        update_description(conn, image_id, raw=raw_desc, enriched=enriched)

        # Re-index in Zvec
        search = get_search()
        search.add(image_id, enriched)

        # Clear story cache for albums containing this photo so regenerated stories pick up the fix
        conn.execute(
            "DELETE FROM story_cache WHERE album_id IN (SELECT album_id FROM album_images WHERE image_id = ?)",
            (image_id,),
        )

        updated_row = get_image_by_id(conn, image_id)
        place = updated_row["place_name"] or ""
        camera = updated_row["camera_model"] or ""
        dt = updated_row["date_taken"] or ""

        return {
            "status": "ok",
            "id": image_id,
            "raw_description": raw_desc,
            "enriched_text": enriched,
            "tags": extract_tags(enriched, place, camera, dt),
        }



# ── Bulk Operations & Location Search ────────────────────

@app.get("/api/locations/search")
def search_locations_endpoint(q: str = Query("", min_length=1), limit: int = Query(8, ge=1, le=50)):
    """Fast offline location search across global places for Google Calendar style location picker."""
    places = search_locations(q, limit=limit)
    return {"query": q, "count": len(places), "results": places}


@app.post("/api/photos/bulk-location")
def bulk_location_endpoint(req: BulkLocationRequest):
    """Assign location to multiple selected photos, update metadata, and re-embed."""
    if not req.image_ids:
        raise HTTPException(400, "No photos selected.")

    search = get_search()
    place_name = req.place_name.strip()
    with get_conn() as conn:
        updated_rows = bulk_update_location(
            conn,
            req.image_ids,
            place_name=place_name,
            latitude=req.latitude,
            longitude=req.longitude,
            city=req.city.strip(),
            region=req.region.strip(),
            country=req.country.strip(),
        )
        for row in updated_rows:
            enriched = enrich_description(
                row["raw_description"] or "",
                {
                    "date_taken": row["date_taken"],
                    "place_name": place_name,
                    "camera_model": row["camera_model"],
                },
            )
            update_description(conn, row["id"], raw=row["raw_description"] or "", enriched=enriched)
            search.add(row["id"], enriched)

    return {
        "status": "ok",
        "updated_count": len(updated_rows),
        "place_name": place_name,
        "latitude": req.latitude,
        "longitude": req.longitude,
    }


@app.post("/api/photos/bulk-album")
def bulk_album_endpoint(req: BulkAlbumRequest):
    """Add multiple photos to an album."""
    if not req.image_ids:
        raise HTTPException(400, "No photos selected.")
    with get_conn() as conn:
        album = get_album_by_id(conn, req.album_id)
        if not album:
            raise HTTPException(404, f"Album {req.album_id} not found.")
        count = bulk_add_photos_to_album(conn, req.album_id, req.image_ids)
    return {
        "status": "ok",
        "album_id": req.album_id,
        "album_name": album["name"],
        "added_count": count,
    }


@app.post("/api/photos/bulk-remove-from-album")
def bulk_remove_from_album_endpoint(req: BulkRemoveAlbumRequest):
    """Remove multiple photos from a specific album."""
    if not req.image_ids:
        raise HTTPException(400, "No photos selected.")
    with get_conn() as conn:
        album = get_album_by_id(conn, req.album_id)
        if not album:
            raise HTTPException(404, f"Album {req.album_id} not found.")
        removed_count = bulk_remove_photos_from_album(conn, req.album_id, req.image_ids)
    return {
        "status": "ok",
        "album_id": req.album_id,
        "album_name": album["name"],
        "removed_count": removed_count,
    }


@app.post("/api/photos/bulk-move-album")
def bulk_move_album_endpoint(req: BulkMoveAlbumRequest):
    """Move multiple photos from source album to target album."""
    if not req.image_ids:
        raise HTTPException(400, "No photos selected.")
    with get_conn() as conn:
        source = get_album_by_id(conn, req.source_album_id)
        target = get_album_by_id(conn, req.target_album_id)
        if not source:
            raise HTTPException(404, f"Source album {req.source_album_id} not found.")
        if not target:
            raise HTTPException(404, f"Target album {req.target_album_id} not found.")
        moved_count = bulk_move_photos_to_album(
            conn, req.source_album_id, req.target_album_id, req.image_ids
        )
    return {
        "status": "ok",
        "source_album_id": req.source_album_id,
        "target_album_id": req.target_album_id,
        "source_name": source["name"],
        "target_name": target["name"],
        "moved_count": moved_count,
    }


@app.post("/api/photos/bulk-delete")
def bulk_delete_endpoint(req: BulkDeleteRequest):
    """Remove selected photos from library database, thumbnails, and vector index (original files preserved)."""
    if not req.image_ids:
        raise HTTPException(400, "No photos selected.")

    with get_conn() as conn:
        paths = delete_images(conn, req.image_ids)

    # Delete cached thumbnails
    for iid in req.image_ids:
        tpath = THUMB_DIR / f"{iid}.jpg"
        if tpath.exists():
            try:
                tpath.unlink()
            except Exception:
                pass

    # Delete from vector index
    get_search().delete(req.image_ids)

    return {"status": "ok", "deleted_count": len(req.image_ids)}


@app.post("/api/photos/bulk-reprocess")
def bulk_reprocess_endpoint(req: BulkReprocessRequest):
    """Queue selected photos for AI vision re-captioning and re-embedding."""
    if not req.image_ids:
        raise HTTPException(400, "No photos selected.")

    target_ids = list(req.image_ids)

    chosen_model = req.vlm_model
    def reprocess_worker():
        from backend.describer import ImageDescriber
        describer = ImageDescriber(model_name=chosen_model)
        describer.load()
        search = get_search()

        with get_conn() as conn:
            placeholders = ",".join("?" for _ in target_ids)
            rows = conn.execute(f"SELECT * FROM images WHERE id IN ({placeholders})", target_ids).fetchall()

        for r in rows:
            try:
                raw_desc = describer.describe(r["file_path"])
                enriched = enrich_description(raw_desc, {
                    "date_taken": r["date_taken"],
                    "place_name": r["place_name"],
                    "camera_model": r["camera_model"],
                })
                with get_conn() as conn:
                    update_description(conn, r["id"], raw=raw_desc, enriched=enriched)
                    mark_embedded(conn, r["id"])
                search.add(r["id"], enriched)
                # Regenerate thumbnail with correct EXIF orientation
                generate_thumbnail(r["file_path"], r["id"], force=True)
            except Exception as e:
                print(f"Error reprocessing image #{r['id']}: {e}")

    t = threading.Thread(target=reprocess_worker, daemon=True)
    t.start()
    return {"status": "queued", "count": len(target_ids), "message": f"Queued {len(target_ids)} photos for AI re-analysis"}


@app.post("/api/photos/regenerate-thumbnails")
def regenerate_thumbnails_endpoint():
    """Regenerate all thumbnails with EXIF auto-transposition so sideways images are upright."""
    with get_conn() as conn:
        records = get_all_image_paths(conn)
    count = regenerate_all_thumbnails(records)
    return {"status": "ok", "regenerated": count, "total": len(records)}


@app.get("/api/thumb/{image_id}")
def get_thumbnail(image_id: int):
    """Serve a thumbnail JPEG."""
    thumb_path = THUMB_DIR / f"{image_id}.jpg"
    if not thumb_path.exists():
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(thumb_path, media_type="image/jpeg")


@app.get("/api/preview/{image_id}")
def get_preview(image_id: int):
    """
    Serve a high-quality web-compatible 2048px JPEG preview.
    Converts HEIC/HEIF/TIFF files into crisp JPEG and caches them on disk.
    Loads instantly (~5ms) after first generation.
    """
    preview_path = PREVIEW_DIR / f"{image_id}.jpg"
    if preview_path.exists() and preview_path.stat().st_size > 0:
        return FileResponse(preview_path, media_type="image/jpeg")

    with get_conn() as conn:
        row = get_image_by_id(conn, image_id)
    if not row:
        raise HTTPException(404, "Image not found")

    fpath = Path(row["file_path"])
    if not fpath.exists():
        raise HTTPException(404, "Original file not found on disk")

    try:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(fpath) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(preview_path, "JPEG", quality=88)
        return FileResponse(preview_path, media_type="image/jpeg")
    except Exception as e:
        thumb_path = THUMB_DIR / f"{image_id}.jpg"
        if thumb_path.exists():
            return FileResponse(thumb_path, media_type="image/jpeg")
        raise HTTPException(500, f"Failed to generate preview: {e}")


@app.get("/api/original/{image_id}")
def get_original(image_id: int, download: bool = False, raw: bool = False):
    """Serve the original image file or a web-compatible preview for HEIC/TIFF."""
    with get_conn() as conn:
        row = get_image_by_id(conn, image_id)
    if row is None:
        raise HTTPException(404, "Image not found")

    fpath = Path(row["file_path"])
    if not fpath.exists():
        raise HTTPException(404, "Original file not found on disk")

    suffix = fpath.suffix.lower()

    # If browser requests a HEIC/HEIF/TIFF without explicit download/raw flag,
    # serve the web-compatible JPEG preview so the browser doesn't show a blank image!
    if suffix in (".heic", ".heif", ".tif", ".tiff") and not (download or raw):
        return get_preview(image_id)

    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".heic": "image/heic", ".heif": "image/heif",
        ".gif": "image/gif", ".bmp": "image/bmp",
        ".tif": "image/tiff", ".tiff": "image/tiff",
    }
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{fpath.name}"'
    return FileResponse(fpath, media_type=media_types.get(suffix, "application/octet-stream"), headers=headers)


class StoryGenerateRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    model: Optional[str] = None


class StoryGroupGenerateRequest(BaseModel):
    album_id: Optional[int] = None
    group_title: str
    group_key: Optional[str] = None
    image_ids: list[int]
    model: Optional[str] = None
    save_to_cache: bool = True


class StorySingleDayRequest(BaseModel):
    day_date: str
    model: Optional[str] = None
    force: bool = True


# Simple progress tracker for story generation
_story_progress: dict = {}


@app.post("/api/story/generate-group")
def generate_group_story_endpoint(req: StoryGroupGenerateRequest):
    """Generate a story narrative for an arbitrary group of photos or section."""
    from backend.storyteller import generate_group_narrative
    from backend.config import STORY_LLM_MODEL
    from backend.db import get_image_by_id, upsert_narrative

    if not req.image_ids:
        raise HTTPException(400, "No photos provided for group story.")

    model = (req.model or "").strip() or STORY_LLM_MODEL

    with get_conn() as conn:
        photos = []
        for img_id in req.image_ids:
            row = get_image_by_id(conn, img_id)
            if row:
                photos.append(dict(row))

    if not photos:
        raise HTTPException(404, "None of the specified photos were found.")

    try:
        narrative = generate_group_narrative(req.group_title, photos, model=model)
    except Exception as e:
        raise HTTPException(500, f"Story generation error: {e}")

    # If associated with an album and group_key / day_date is provided, cache it
    if req.album_id and req.save_to_cache and req.group_key:
        import json
        with get_conn() as conn:
            upsert_narrative(
                conn,
                album_id=req.album_id,
                day_date=req.group_key,
                narrative=narrative,
                model_used=model,
                photo_ids=json.dumps([p["id"] for p in photos]),
            )

    return {
        "status": "ok",
        "narrative": narrative,
        "model_used": model,
        "group_title": req.group_title,
        "group_key": req.group_key,
        "photo_ids": [p["id"] for p in photos],
        "photo_count": len(photos),
        "album_id": req.album_id,
    }


@app.post("/api/albums/{album_id}/story/day/generate")
def generate_single_day_story_endpoint(album_id: int, req: StorySingleDayRequest):
    """Generate or regenerate story for a single day in an album."""
    from backend.storyteller import generate_single_day_story
    from backend.config import STORY_LLM_MODEL

    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
    if not album:
        raise HTTPException(404, f"Album {album_id} not found.")

    model = (req.model or "").strip() or STORY_LLM_MODEL

    try:
        res = generate_single_day_story(
            album_id=album_id,
            day_date=req.day_date,
            model=model,
            force=req.force,
        )
        return {"status": "ok", "album_id": album_id, **res}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Story generation failed: {e}")


@app.delete("/api/albums/{album_id}/story/day/{day_date}")
def delete_single_day_story_endpoint(album_id: int, day_date: str):
    """Delete a single cached day narrative from an album."""
    from backend.db import delete_day_narrative

    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")
        deleted = delete_day_narrative(conn, album_id, day_date)

    return {"deleted": deleted, "album_id": album_id, "day_date": day_date}


@app.post("/api/albums/{album_id}/story/generate")
def generate_story_endpoint(album_id: int, req: StoryGenerateRequest):
    """Kick off story timeline generation for an album (runs in background)."""
    from backend.storyteller import generate_album_story
    from backend.config import STORY_LLM_MODEL

    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
    if not album:
        raise HTTPException(404, f"Album {album_id} not found.")

    model = (req.model or "").strip() or STORY_LLM_MODEL
    progress_key = f"story_{album_id}"

    # Check if already generating
    if progress_key in _story_progress and _story_progress[progress_key].get("status") == "generating":
        return {"status": "already_generating", "progress": _story_progress[progress_key]}

    _story_progress[progress_key] = {
        "status": "generating",
        "current": 0,
        "total": 0,
        "current_day": "",
        "album_id": album_id,
    }

    def _run():
        try:
            def progress_cb(idx, total, day_date):
                _story_progress[progress_key].update({
                    "current": idx,
                    "total": total,
                    "current_day": day_date,
                })

            results = generate_album_story(
                album_id,
                start_date=req.start_date,
                end_date=req.end_date,
                model=model,
                progress_callback=progress_cb,
            )
            _story_progress[progress_key] = {
                "status": "complete",
                "current": len(results),
                "total": len(results),
                "current_day": "",
                "album_id": album_id,
            }
        except Exception as e:
            _story_progress[progress_key] = {
                "status": "error",
                "error": str(e),
                "album_id": album_id,
            }

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {"status": "started", "progress": _story_progress[progress_key]}


@app.get("/api/albums/{album_id}/story/progress")
def story_progress_endpoint(album_id: int):
    """Check story generation progress."""
    progress_key = f"story_{album_id}"
    progress = _story_progress.get(progress_key)
    if not progress:
        return {"status": "idle"}
    return progress


@app.get("/api/albums/{album_id}/story")
def get_story_endpoint(album_id: int):
    """Fetch the generated story timeline from cache."""
    from backend.db import get_album_story

    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")

        stories = get_album_story(conn, album_id)

    # Parse photo_ids JSON and attach thumbnail URLs
    import json
    days = []
    for s in stories:
        try:
            photo_ids = json.loads(s["photo_ids"])
        except (json.JSONDecodeError, TypeError):
            photo_ids = []
        days.append({
            "day_date": s["day_date"],
            "narrative": s["narrative"],
            "model_used": s["model_used"],
            "photo_ids": photo_ids,
            "photo_count": len(photo_ids),
            "created_at": s["created_at"],
        })

    return {
        "album_id": album_id,
        "album_name": album["name"],
        "days": days,
        "total_days": len(days),
    }


@app.delete("/api/albums/{album_id}/story")
def clear_story_endpoint(album_id: int):
    """Clear cached story for regeneration."""
    from backend.db import clear_story_cache

    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")
        deleted = clear_story_cache(conn, album_id)

    # Clear progress too
    _story_progress.pop(f"story_{album_id}", None)

    return {"deleted": deleted, "album_id": album_id}


# ── Album Notes Endpoints ────────────────────────────────

@app.get("/api/albums/{album_id}/notes")
def get_album_notes_endpoint(album_id: int):
    """List all saved notes for an album."""
    from backend.db import get_album_notes
    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")
        notes = get_album_notes(conn, album_id)
    return {"album_id": album_id, "album_name": album["name"], "count": len(notes), "notes": notes}


@app.post("/api/albums/{album_id}/notes")
def create_album_note_endpoint(album_id: int, req: CreateNoteRequest):
    """Create a new note for an album."""
    from backend.db import create_album_note
    if not req.title.strip() and not req.content.strip():
        raise HTTPException(400, "Note title or content cannot be empty.")
    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")
        note = create_album_note(
            conn,
            album_id=album_id,
            day_date=req.day_date or "General",
            title=req.title or "Album Note",
            content=req.content,
        )
    return {"status": "ok", "note": note}


@app.post("/api/albums/{album_id}/story/save-note")
def save_story_as_note_endpoint(album_id: int, req: SaveStoryNoteRequest):
    """Save or update a story day narrative as an editable album note."""
    from backend.db import save_or_update_story_note
    if not req.content.strip():
        raise HTTPException(400, "Note content cannot be empty.")
    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")
        note = save_or_update_story_note(
            conn,
            album_id=album_id,
            day_date=req.day_date,
            title=req.title,
            content=req.content,
        )
    return {"status": "ok", "note": note}


@app.get("/api/notes/search")
def search_notes_endpoint(q: str = Query("", min_length=0), limit: int = Query(50, ge=1, le=100)):
    """Search notes across all albums by title, content, date, or album name."""
    from backend.db import search_all_notes
    with get_conn() as conn:
        notes = search_all_notes(conn, query=q, limit=limit)
    return {"query": q, "count": len(notes), "notes": notes}


@app.get("/api/notes/{note_id}")
def get_single_note_endpoint(note_id: int):
    """Get a single note by ID."""
    from backend.db import get_note_by_id
    with get_conn() as conn:
        note = get_note_by_id(conn, note_id)
    if not note:
        raise HTTPException(404, f"Note {note_id} not found.")
    return {"note": note}


@app.put("/api/notes/{note_id}")
def update_note_endpoint(note_id: int, req: UpdateNoteRequest):
    """Update title, content, and date of an existing note."""
    from backend.db import update_album_note
    if not req.title.strip() and not req.content.strip():
        raise HTTPException(400, "Note title or content cannot be empty.")
    with get_conn() as conn:
        note = update_album_note(
            conn,
            note_id=note_id,
            title=req.title,
            content=req.content,
            day_date=req.day_date,
        )
    if not note:
        raise HTTPException(404, f"Note {note_id} not found.")
    return {"status": "ok", "note": note}


@app.delete("/api/notes/{note_id}")
def delete_note_endpoint(note_id: int):
    """Delete a note."""
    from backend.db import delete_album_note
    with get_conn() as conn:
        deleted = delete_album_note(conn, note_id)
    if not deleted:
        raise HTTPException(404, f"Note {note_id} not found.")
    return {"status": "ok", "deleted": True}


# ── People & Faces API ───────────────────────────────────

class FaceCreateRequest(BaseModel):
    box_x: float
    box_y: float
    box_w: float
    box_h: float
    confidence: float = 1.0
    is_pet: bool = False
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    relationship: Optional[str] = "Friend"


class FaceUpdateRequest(BaseModel):
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    relationship: Optional[str] = "Friend"
    is_pet: Optional[bool] = None


class PersonCreateRequest(BaseModel):
    name: str
    relationship: Optional[str] = "Friend"
    notes: Optional[str] = ""
    avatar_face_id: Optional[int] = None


class PersonUpdateRequest(BaseModel):
    name: Optional[str] = None
    relationship: Optional[str] = None
    notes: Optional[str] = None
    avatar_face_id: Optional[int] = None


class BatchTagRequest(BaseModel):
    face_ids: list[int]


_face_scan_state = {
    "is_running": False,
    "current": 0,
    "total": 0,
    "faces_found": 0,
    "error": None,
}


@app.get("/api/images/{image_id}/faces")
def get_image_faces(image_id: int):
    """Return all faces for an image, including matching suggestions for untagged faces."""
    with get_conn() as conn:
        img_row = get_image_by_id(conn, image_id)
        faces = get_faces_for_image(conn, image_id)
        known_embeddings = get_known_face_embeddings(conn)
        all_people = {p["id"]: p for p in get_all_people(conn)}
        rejections = conn.execute(
            "SELECT face_id, person_id FROM face_rejections WHERE face_id IN (SELECT id FROM faces WHERE image_id = ?)",
            (image_id,),
        ).fetchall()
        rejected_set = {(r["face_id"], r["person_id"]) for r in rejections}

    # Track people already tagged on this image to avoid redundant suggestions of the same person
    already_tagged_people = {f["person_id"] for f in faces if f.get("person_id")}

    results = []
    for f in faces:
        face_dict = dict(f)
        face_id = f["id"]
        if not f.get("person_id") and f.get("has_embedding"):
            with get_conn() as conn:
                rec = conn.execute("SELECT embedding FROM faces WHERE id = ?", (face_id,)).fetchone()
                emb = rec["embedding"] if rec else None
            if emb:
                # Exclude candidates that were rejected for this specific face or already tagged on this photo
                valid_known = [
                    (fid, pid, e)
                    for (fid, pid, e) in known_embeddings
                    if (face_id, pid) not in rejected_set and pid not in already_tagged_people
                ]
                if valid_known:
                    best_pid, score = match_face_embedding(emb, valid_known)
                    if best_pid and best_pid in all_people:
                        matched_person = all_people[best_pid]
                        face_dict["suggestion"] = {
                            "person_id": best_pid,
                            "name": matched_person["name"],
                            "relationship": matched_person["relationship"],
                            "similarity": score,
                        }
        results.append(face_dict)

    faces_scanned = bool(img_row["faces_scanned"]) if (img_row and "faces_scanned" in img_row.keys()) else False
    return {"status": "ok", "faces": results, "faces_scanned": faces_scanned}


@app.post("/api/images/{image_id}/faces/detect")
def detect_image_faces(image_id: int, auto_tag: bool = True):
    """
    Run YuNet detection + SFace embedding extraction on an image on-demand.
    Saves new faces to the database. If auto_tag=True, automatically assigns
    high-confidence matches.
    """
    with get_conn() as conn:
        img_row = get_image_by_id(conn, image_id)
    if not img_row:
        raise HTTPException(404, "Image not found")

    file_path = Path(img_row["file_path"])
    if not file_path.exists():
        raise HTTPException(404, "Original image file not found on disk")

    detected = detect_and_embed_faces(file_path)

    with get_conn() as conn:
        conn.execute("UPDATE images SET faces_scanned = 1 WHERE id = ?", (image_id,))
        existing_faces = get_faces_for_image(conn, image_id)
        assigned_person_ids = {ef["person_id"] for ef in existing_faces if ef.get("person_id")}
        known_embeddings = get_known_face_embeddings(conn)

        for d in detected:
            is_dup = False
            for ef in existing_faces:
                if calculate_box_iou(ef, d) >= 0.45:
                    is_dup = True
                    break
            if is_dup:
                continue

            person_id = None
            if auto_tag and d.get("embedding"):
                best_pid, score = match_face_embedding(d["embedding"], known_embeddings)
                if best_pid and best_pid not in assigned_person_ids:
                    person_id = best_pid
                    assigned_person_ids.add(best_pid)

            insert_face(
                conn,
                image_id=image_id,
                box_x=d["box_x"],
                box_y=d["box_y"],
                box_w=d["box_w"],
                box_h=d["box_h"],
                confidence=d["confidence"],
                embedding=d["embedding"],
                person_id=person_id,
                is_pet=1 if d.get("is_pet") else 0,
            )

    return get_image_faces(image_id)


@app.post("/api/images/{image_id}/faces")
def create_face(image_id: int, req: FaceCreateRequest):
    """Manually add a face or pet tag box."""
    with get_conn() as conn:
        img_row = get_image_by_id(conn, image_id)
        if not img_row:
            raise HTTPException(404, "Image not found")

        person_id = req.person_id
        if req.person_name and req.person_name.strip():
            person_id = upsert_person(
                conn,
                name=req.person_name.strip(),
                relationship=req.relationship or "Friend",
            )

        fid = insert_face(
            conn,
            image_id=image_id,
            box_x=max(0.0, min(1.0, req.box_x)),
            box_y=max(0.0, min(1.0, req.box_y)),
            box_w=max(0.01, min(1.0, req.box_w)),
            box_h=max(0.01, min(1.0, req.box_h)),
            confidence=req.confidence,
            embedding=None,
            person_id=person_id,
            is_pet=1 if req.is_pet else 0,
        )

        if person_id:
            p = get_person_by_id(conn, person_id)
            if p and not p.get("avatar_face_id"):
                update_person(conn, person_id, avatar_face_id=fid)

        face = get_face_by_id(conn, fid)

    return {"status": "ok", "face": face}


@app.put("/api/faces/{face_id}")
def update_face_endpoint(face_id: int, req: FaceUpdateRequest):
    """Assign or change a person on an existing face."""
    with get_conn() as conn:
        face = get_face_by_id(conn, face_id)
        if not face:
            raise HTTPException(404, "Face not found")

        person_id = req.person_id
        if req.person_name and req.person_name.strip():
            person_id = upsert_person(
                conn,
                name=req.person_name.strip(),
                relationship=req.relationship or "Friend",
            )

        update_face_person(conn, face_id, person_id)

        if req.is_pet is not None:
            conn.execute("UPDATE faces SET is_pet = ? WHERE id = ?", (1 if req.is_pet else 0, face_id))

        if person_id:
            p = get_person_by_id(conn, person_id)
            if p and not p.get("avatar_face_id"):
                update_person(conn, person_id, avatar_face_id=face_id)

        updated = get_face_by_id(conn, face_id)

    return {"status": "ok", "face": updated}


@app.delete("/api/faces/{face_id}")
def delete_face_endpoint(face_id: int):
    """Delete a face bounding box and tag."""
    with get_conn() as conn:
        deleted = delete_face(conn, face_id)
    if not deleted:
        raise HTTPException(404, "Face not found")
    return {"status": "ok", "deleted": True}


@app.get("/api/faces/thumb/{face_id}")
def get_face_thumbnail(face_id: int):
    """Serve cropped avatar thumbnail for a face."""
    with get_conn() as conn:
        face = get_face_by_id(conn, face_id)
    if not face:
        raise HTTPException(404, "Face not found")

    cached_thumb = FACES_THUMB_DIR / f"{face_id}.jpg"
    if cached_thumb.exists():
        return FileResponse(cached_thumb, media_type="image/jpeg")

    box = {
        "box_x": face["box_x"],
        "box_y": face["box_y"],
        "box_w": face["box_w"],
        "box_h": face["box_h"],
    }
    jpeg_bytes = crop_face_thumbnail(face["file_path"], box, output_size=160)
    if not jpeg_bytes:
        raise HTTPException(500, "Failed to crop face thumbnail")

    try:
        with open(cached_thumb, "wb") as f:
            f.write(jpeg_bytes)
    except Exception:
        pass

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/people")
def list_people():
    """Return all known people and pets with stats."""
    with get_conn() as conn:
        people = get_all_people(conn)
    return {"status": "ok", "people": people}


@app.post("/api/people")
def create_person(req: PersonCreateRequest):
    """Create or update a person/pet record."""
    with get_conn() as conn:
        pid = upsert_person(
            conn,
            name=req.name,
            relationship=req.relationship or "Friend",
            notes=req.notes or "",
            avatar_face_id=req.avatar_face_id,
        )
        person = get_person_by_id(conn, pid)
    return {"status": "ok", "person": person}


@app.get("/api/people/{person_id}")
def get_person_endpoint(person_id: int):
    """Get person profile and stats."""
    with get_conn() as conn:
        person = get_person_by_id(conn, person_id)
    if not person:
        raise HTTPException(404, "Person not found")
    return {"status": "ok", "person": person}


@app.put("/api/people/{person_id}")
def update_person_endpoint(person_id: int, req: PersonUpdateRequest):
    """Update person details (name, relationship, notes, avatar)."""
    with get_conn() as conn:
        updated = update_person(
            conn,
            person_id=person_id,
            name=req.name,
            relationship=req.relationship,
            notes=req.notes,
            avatar_face_id=req.avatar_face_id,
        )
    if not updated:
        raise HTTPException(404, "Person not found")
    return {"status": "ok", "person": updated}


@app.delete("/api/people/{person_id}")
def delete_person_endpoint(person_id: int):
    """Delete a person profile. Faces remain with person_id=NULL."""
    with get_conn() as conn:
        deleted = delete_person(conn, person_id)
    if not deleted:
        raise HTTPException(404, "Person not found")
    return {"status": "ok", "deleted": True}


@app.get("/api/people/{person_id}/photos")
def get_person_photos(person_id: int):
    """Return all photos featuring this person or pet."""
    with get_conn() as conn:
        person = get_person_by_id(conn, person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        photos = get_photos_for_person(conn, person_id)
    return {"status": "ok", "person": person, "photos": photos, "count": len(photos)}


@app.post("/api/people/{person_id}/suggest-matches")
def suggest_person_matches(person_id: int):
    """Find all untagged faces across the library that match this person's face embeddings."""
    with get_conn() as conn:
        person = get_person_by_id(conn, person_id)
        if not person:
            raise HTTPException(404, "Person not found")

        person_rows = conn.execute(
            "SELECT embedding FROM faces WHERE person_id = ? AND embedding IS NOT NULL",
            (person_id,),
        ).fetchall()
        if not person_rows:
            return {"status": "ok", "suggestions": [], "message": "No face embeddings available for this person."}

        person_embeddings = [r["embedding"] for r in person_rows]
        untagged = get_untagged_faces_with_embeddings(conn, limit=200, exclude_person_id=person_id)

    from backend.faces import get_recognizer
    import numpy as np
    import cv2

    recognizer = get_recognizer()
    p_feats = [np.frombuffer(e, dtype=np.float32).reshape(1, -1) for e in person_embeddings]

    suggestions = []
    for uf in untagged:
        u_feat = np.frombuffer(uf["embedding"], dtype=np.float32).reshape(1, -1)
        scores = [float(recognizer.match(pf, u_feat, cv2.FaceRecognizerSF_FR_COSINE)) for pf in p_feats]
        max_s = max(scores) if scores else 0.0
        if max_s >= 0.363:
            suggestions.append({
                "face_id": uf["id"],
                "image_id": uf["image_id"],
                "file_path": uf["file_path"],
                "similarity": round(max_s, 3),
                "box": {
                    "box_x": uf["box_x"],
                    "box_y": uf["box_y"],
                    "box_w": uf["box_w"],
                    "box_h": uf["box_h"],
                },
            })

    suggestions.sort(key=lambda x: x["similarity"], reverse=True)
    return {"status": "ok", "person": person, "suggestions": suggestions, "count": len(suggestions)}


@app.post("/api/people/{person_id}/batch-tag")
def batch_tag_person(person_id: int, req: BatchTagRequest):
    """Assign multiple faces to this person in 1 click."""
    with get_conn() as conn:
        person = get_person_by_id(conn, person_id)
        if not person:
            raise HTTPException(404, "Person not found")

        count = 0
        for fid in req.face_ids:
            if update_face_person(conn, fid, person_id):
                count += 1

    return {"status": "ok", "tagged_count": count}


@app.post("/api/people/{person_id}/reject-match/{face_id}")
def reject_match_person(person_id: int, face_id: int):
    """Mark a face as not matching this person so it won't be suggested again."""
    from backend.db import reject_face_match
    with get_conn() as conn:
        person = get_person_by_id(conn, person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        reject_face_match(conn, face_id, person_id)
    return {"status": "ok", "message": f"Rejected face match for {person['name']}"}


@app.post("/api/people/{person_id}/batch-reject")
def batch_reject_person_matches(person_id: int, req: BatchTagRequest):
    """Reject multiple face matches for this person."""
    from backend.db import batch_reject_face_matches
    with get_conn() as conn:
        person = get_person_by_id(conn, person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        count = batch_reject_face_matches(conn, req.face_ids, person_id)
    return {"status": "ok", "rejected_count": count}


@app.post("/api/faces/batch-delete")
def batch_delete_faces_endpoint(req: BatchTagRequest):
    """Delete multiple face boxes completely (e.g. false detections / not a face)."""
    from backend.db import batch_delete_faces
    with get_conn() as conn:
        count = batch_delete_faces(conn, req.face_ids)
    return {"status": "ok", "deleted_count": count}


@app.get("/api/people/{person_id}/faces")
def get_person_faces(person_id: int):
    """Return all detected face variations / occurrences for this person across the library."""
    with get_conn() as conn:
        person = get_person_by_id(conn, person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        faces = get_faces_for_person(conn, person_id)
    return {"status": "ok", "person": person, "faces": faces, "count": len(faces)}


@app.post("/api/people/{person_id}/unlink-face/{face_id}")
def unlink_person_face(person_id: int, face_id: int):
    """Remove a false detection / tag from a person, releasing the face back to untagged."""
    with get_conn() as conn:
        person = get_person_by_id(conn, person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        ok = unlink_face_from_person(conn, person_id, face_id)
        if not ok:
            raise HTTPException(404, "Face not found or not assigned to this person")
        updated_person = get_person_by_id(conn, person_id)
    return {"status": "ok", "message": "Face unlinked from person successfully", "person": updated_person}


@app.post("/api/people/{person_id}/set-avatar/{face_id}")
def set_person_avatar(person_id: int, face_id: int):
    """Set a specific face variation as the profile avatar for this person."""
    with get_conn() as conn:
        person = get_person_by_id(conn, person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        face = get_face_by_id(conn, face_id)
        if not face:
            raise HTTPException(404, "Face not found")
        update_person(conn, person_id, avatar_face_id=face_id)
        updated = get_person_by_id(conn, person_id)
    return {"status": "ok", "person": updated}


def _run_library_face_scan():
    """Background worker to scan all library images for faces."""
    global _face_scan_state
    _face_scan_state["is_running"] = True
    _face_scan_state["faces_found"] = 0
    _face_scan_state["error"] = None

    try:
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT i.id, i.file_path 
                   FROM images i
                   WHERE i.faces_scanned = 0 OR i.faces_scanned IS NULL"""
            ).fetchall()

        total = len(rows)
        _face_scan_state["total"] = total
        _face_scan_state["current"] = 0

        for idx, r in enumerate(rows):
            img_id = r["id"]
            fpath = Path(r["file_path"])
            if fpath.exists():
                try:
                    detected = detect_and_embed_faces(fpath)
                    with get_conn() as conn:
                        conn.execute("UPDATE images SET faces_scanned = 1 WHERE id = ?", (img_id,))
                        if detected:
                            existing_faces = get_faces_for_image(conn, img_id)
                            assigned_person_ids = {ef["person_id"] for ef in existing_faces if ef.get("person_id")}
                            known_embeddings = get_known_face_embeddings(conn)
                            for d in detected:
                                is_dup = False
                                for ef in existing_faces:
                                    if calculate_box_iou(ef, d) >= 0.45:
                                        is_dup = True
                                        break
                                if is_dup:
                                    continue

                                person_id = None
                                if d.get("embedding"):
                                    best_pid, score = match_face_embedding(d["embedding"], known_embeddings)
                                    if best_pid and best_pid not in assigned_person_ids:
                                        person_id = best_pid
                                        assigned_person_ids.add(best_pid)
                                insert_face(
                                    conn,
                                    image_id=img_id,
                                    box_x=d["box_x"],
                                    box_y=d["box_y"],
                                    box_w=d["box_w"],
                                    box_h=d["box_h"],
                                    confidence=d["confidence"],
                                    embedding=d["embedding"],
                                    person_id=person_id,
                                    is_pet=1 if d.get("is_pet") else 0,
                                )
                                _face_scan_state["faces_found"] += 1
                except Exception as e:
                    print(f"[FaceScan] Error scanning image {img_id}: {e}")

            _face_scan_state["current"] = idx + 1
    except Exception as e:
        _face_scan_state["error"] = str(e)
    finally:
        _face_scan_state["is_running"] = False


@app.post("/api/faces/scan-library")
def start_face_scan():
    """Trigger background library face scan."""
    global _face_scan_state
    if _face_scan_state["is_running"]:
        return {"status": "running", "message": "Scan is already in progress."}

    t = threading.Thread(target=_run_library_face_scan, daemon=True)
    t.start()
    return {"status": "started", "message": "Library face scan initiated in background."}


@app.get("/api/faces/scan-status")
def get_face_scan_status():
    """Return status and progress of the background library face scan."""
    return {"status": "ok", **_face_scan_state}



class CollageSaveRequest(BaseModel):
    image_data: str
    title: Optional[str] = "Photo Collage"
    album_id: Optional[int] = None
    aspect_ratio: Optional[str] = "1:1"
    layout: Optional[str] = "grid"


@app.post("/api/collage/save")
def save_collage(req: CollageSaveRequest):
    """Save a user-generated collage image to disk and index in database."""
    try:
        import base64
        import hashlib
        from datetime import datetime
        import uuid
        from backend.db import upsert_image, update_metadata, add_photo_to_album, get_image_by_id
        from backend.scanner import make_thumbnail

        data = req.image_data
        if "," in data:
            data = data.split(",", 1)[1]
        img_bytes = base64.b64decode(data)

        collages_dir = DATA_DIR / "collages"
        collages_dir.mkdir(parents=True, exist_ok=True)

        file_hash = hashlib.sha256(img_bytes).hexdigest()
        filename = f"collage_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
        file_path = collages_dir / filename
        file_path.write_bytes(img_bytes)

        file_size = len(img_bytes)
        now_iso = datetime.now().isoformat()

        with get_conn() as conn:
            image_id = upsert_image(conn, str(file_path.resolve()), file_hash, file_size)
            meta = {
                "date_taken": now_iso,
                "camera_make": "PixelMemory",
                "camera_model": f"Collage Studio ({req.layout} · {req.aspect_ratio})",
                "orientation": 1,
            }
            update_metadata(conn, image_id, **meta)

            desc = f"Custom photo collage created in PixelMemory Collage Studio. Layout: {req.layout}, aspect ratio: {req.aspect_ratio}."
            if req.title:
                desc = f"Photo collage titled '{req.title}'. {desc}"
            conn.execute(
                "UPDATE images SET raw_description = ?, enriched_text = ?, description_done = 1 WHERE id = ?",
                (desc, desc, image_id)
            )

            if req.album_id:
                try:
                    add_photo_to_album(conn, req.album_id, image_id)
                except Exception:
                    pass

        # Generate thumbnail
        try:
            make_thumbnail(file_path, image_id)
        except Exception:
            pass

        saved_img = get_image_by_id(image_id)
        return {
            "status": "ok",
            "message": "Collage saved successfully",
            "image": dict(saved_img) if saved_img else {"id": image_id}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save collage: {str(e)}")



# ── Frontend ─────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=FileResponse)
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html")
    return HTMLResponse("<h1>PixelMemory</h1><p>Frontend not found. Place index.html in /frontend/</p>")


# Mount static assets if they exist
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Entry point ──────────────────────────────────────────

def main():
    import uvicorn
    init_db()
    print(f"\nPixelMemory server starting at http://localhost:{PORT}")
    print(f"Search {get_search().count} embedded images\n")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
