"""
FastAPI server: search API + thumbnail serving + static frontend.

Run:
    python -m backend.server
    # or: uvicorn backend.server:app --host 0.0.0.0 --port 8642
"""

from typing import Optional
import re
import threading
from pathlib import Path
from pydantic import BaseModel

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import THUMB_DIR, HOST, PORT, SEARCH_TOP_K
from backend.db import (
    init_db, get_conn, get_image_by_id, get_stats,
    create_album, get_all_albums, get_album_by_id, rename_album,
    add_photo_to_album, remove_photo_from_album,
    get_album_image_ids, get_photo_albums, delete_album,
    bulk_add_photos_to_album, bulk_remove_photos_from_album, bulk_move_photos_to_album,
    delete_images, bulk_update_location,
    get_all_image_paths, update_description, mark_embedded,
)
from backend.search import SemanticSearch
from backend.ingest import tracker, queue_manager, start_background_import
from backend.metadata import search_locations, regenerate_all_thumbnails, generate_thumbnail
from backend.describer import enrich_description


class ImportRequest(BaseModel):
    folder_path: str
    skip_describe: bool = False
    vlm_model: Optional[str] = None


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


app = FastAPI(title="PixelMemory", version="1.0.0")

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

    res = queue_manager.enqueue(str(fpath), skip_describe=req.skip_describe, vlm_model=req.vlm_model)
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
    Rescan an existing folder for newly added photos.
    Only newly added photos will have EXIF extracted, AI vision descriptions generated, and vectors embedded.
    All existing photos are preserved.
    """
    fpath = Path(req.folder_path).resolve()
    if not fpath.exists() or not fpath.is_dir():
        raise HTTPException(400, f"Invalid folder directory: '{req.folder_path}' does not exist on disk.")

    res = queue_manager.enqueue(str(fpath), skip_describe=req.skip_describe, vlm_model=req.vlm_model)
    res["is_rescan"] = True
    res["folder_path"] = str(fpath)
    if res["status"] == "started":
        res["message"] = f"Rescanning '{fpath.name}' for additional photos..."
    elif res["status"] == "queued":
        res["message"] = f"Added '{fpath.name}' rescan to queue (position #{res['queue_position']})"
    return res


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


@app.get("/api/original/{image_id}")
def get_original(image_id: int):
    """Serve the original image file."""
    with get_conn() as conn:
        row = get_image_by_id(conn, image_id)
    if row is None:
        raise HTTPException(404, "Image not found")

    fpath = Path(row["file_path"])
    if not fpath.exists():
        raise HTTPException(404, "Original file not found on disk")

    suffix = fpath.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".heic": "image/heic", ".heif": "image/heif",
        ".gif": "image/gif", ".bmp": "image/bmp",
        ".tif": "image/tiff", ".tiff": "image/tiff",
    }
    return FileResponse(fpath, media_type=media_types.get(suffix, "application/octet-stream"))


# ── Story Timeline ───────────────────────────────────────

class StoryGenerateRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    model: Optional[str] = None


# Simple progress tracker for story generation
_story_progress: dict = {}


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
