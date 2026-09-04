"""
FastAPI server: search API + thumbnail serving + static frontend.

Run:
    python -m backend.server
    # or: uvicorn backend.server:app --host 0.0.0.0 --port 8642
"""

import re
from pathlib import Path
from pydantic import BaseModel

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import THUMB_DIR, HOST, PORT, SEARCH_TOP_K
from backend.db import (
    init_db, get_conn, get_image_by_id, get_stats,
    create_album, get_all_albums, get_album_by_id,
    add_photo_to_album, remove_photo_from_album,
    get_album_image_ids, get_photo_albums, delete_album,
)
from backend.search import SemanticSearch
from backend.ingest import tracker, queue_manager, start_background_import


class ImportRequest(BaseModel):
    folder_path: str
    skip_describe: bool = False


class CreateAlbumRequest(BaseModel):
    name: str
    description: str = ""


class AddPhotoRequest(BaseModel):
    image_id: int

app = FastAPI(title="PixelMemory", version="0.1.0")

# Lazy-loaded search engine
_search: SemanticSearch | None = None


def get_search() -> SemanticSearch:
    global _search
    if _search is None:
        _search = SemanticSearch()
    return _search


def extract_tags(text: str = "", place: str = "", camera: str = "", date_taken: str = "") -> list[str]:
    """Dynamically derive tags from real metadata without hardcoded keyword lists."""
    tags = []
    if place:
        for part in place.split(","):
            part_clean = re.sub(r"[^\w]", "", part.strip())
            if part_clean and len(part_clean) > 2 and part_clean.lower() not in {"us", "usa", "the"}:
                tags.append(f"#{part_clean}")
                break
    if camera:
        cleaned_cam = camera.replace("samsung", "").replace("Apple", "").strip()
        cleaned_cam = re.sub(r"[^\w]", "", cleaned_cam)
        if cleaned_cam:
            tags.append(f"#{cleaned_cam}")
    if date_taken:
        year = date_taken[:4]
        if year.isdigit():
            tags.append(f"#{year}")
    if not tags:
        tags = ["#Photo"]
    return tags[:4]


# ── API routes ───────────────────────────────────────────

@app.get("/api/photos")
def list_all_photos(limit: int = Query(500, ge=1, le=5000), offset: int = Query(0, ge=0)):
    """Return all indexed photos in the library ordered chronologically."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM images 
            ORDER BY COALESCE(date_taken, '') DESC, id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()

        total_row = conn.execute("SELECT COUNT(*) as total FROM images").fetchone()
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
):
    """Semantic search over image descriptions with hybrid scoring and relevance filtering."""
    q_str = q.strip() if q else ""
    if not q_str:
        return list_all_photos(limit=top_k)

    engine = get_search()
    hits = engine.query(q_str, top_k=top_k, filter_mode=filter_mode)

    results = []
    with get_conn() as conn:
        for hit in hits:
            row = get_image_by_id(conn, hit["image_id"])
            if row is None:
                continue
            enriched = hit["enriched_text"] or ""
            place = row["place_name"] or ""
            camera = row["camera_model"] or ""
            dt = row["date_taken"] or ""
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

    from backend.describer import is_ollama_ready
    from backend.config import OLLAMA_HOST, OLLAMA_MODEL

    ollama_ok = is_ollama_ready()
    vlm_label = f"Ollama · {OLLAMA_MODEL}" if ollama_ok else "Moondream2 (PyTorch)"

    with get_conn() as conn:
        stats = get_stats(conn)

    search = get_search()
    return {
        "cuda_available": cuda_avail,
        "device_name": hw_name,
        "vram_gb": vram,
        "vlm_provider": "Ollama" if ollama_ok else "PyTorch",
        "vlm_model": vlm_label,
        "ollama_ready": ollama_ok,
        "ollama_host": OLLAMA_HOST,
        "embedding_model": "all-MiniLM-L6-v2",
        "total_images": stats.get("total", 0),
        "total_vectors": search.count,
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
    try:
        search._client.delete_collection("image_descriptions")
        search._collection = search._client.get_or_create_collection(
            name="image_descriptions",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        pass
    return {"status": "ok", "message": "Database and vector index cleared"}


# ── Import Pipeline Routes ───────────────────────────────

@app.post("/api/import/start")
def start_import_endpoint(req: ImportRequest):
    """Start directory scan and AI ingestion or queue it if already running."""
    fpath = Path(req.folder_path).resolve()
    if not fpath.exists() or not fpath.is_dir():
        raise HTTPException(400, f"Invalid folder directory: '{req.folder_path}' does not exist on disk.")

    res = queue_manager.enqueue(str(fpath), skip_describe=req.skip_describe)
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
def get_album_endpoint(album_id: int):
    """Get album metadata and all photos in this album."""
    with get_conn() as conn:
        album = get_album_by_id(conn, album_id)
        if not album:
            raise HTTPException(404, f"Album {album_id} not found.")

        image_ids = get_album_image_ids(conn, album_id)
        photos = []
        for img_id in image_ids:
            row = get_image_by_id(conn, img_id)
            if row:
                enriched = row["enriched_text"] or ""
                place = row["place_name"] or ""
                camera = row["camera_model"] or ""
                dt = row["date_taken"] or ""
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
