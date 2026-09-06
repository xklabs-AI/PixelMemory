"""SQLite database for image metadata and descriptions."""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from backend.config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT    NOT NULL UNIQUE,
    file_hash       TEXT    NOT NULL,
    file_size       INTEGER NOT NULL,

    -- EXIF metadata (nullable — not all images have EXIF)
    date_taken      TEXT,               -- ISO 8601
    latitude        REAL,
    longitude       REAL,
    camera_make     TEXT,
    camera_model    TEXT,
    orientation     INTEGER,

    -- Reverse-geocoded location
    city            TEXT,
    region          TEXT,
    country         TEXT,
    place_name      TEXT,               -- most specific name available

    -- VLM output
    raw_description TEXT,               -- direct model output
    enriched_text   TEXT,               -- metadata + description combined

    -- Pipeline state
    metadata_done   INTEGER DEFAULT 0,
    description_done INTEGER DEFAULT 0,
    embedded        INTEGER DEFAULT 0,

    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_images_hash ON images(file_hash);
CREATE INDEX IF NOT EXISTS idx_images_date ON images(date_taken);
CREATE INDEX IF NOT EXISTS idx_images_location ON images(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_images_pipeline ON images(metadata_done, description_done, embedded);

-- Albums
CREATE TABLE IF NOT EXISTS albums (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    description TEXT    DEFAULT '',
    created_at  TEXT    DEFAULT (datetime('now')),
    updated_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS album_images (
    album_id    INTEGER NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    added_at    TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (album_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_album_images_album ON album_images(album_id);
CREATE INDEX IF NOT EXISTS idx_album_images_image ON album_images(image_id);

-- Story Timeline cache
CREATE TABLE IF NOT EXISTS story_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    album_id    INTEGER NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    day_date    TEXT    NOT NULL,
    narrative   TEXT    NOT NULL,
    model_used  TEXT    NOT NULL,
    photo_ids   TEXT    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now')),
    UNIQUE(album_id, day_date)
);

CREATE INDEX IF NOT EXISTS idx_story_cache_album ON story_cache(album_id);

-- Album Notes (Editable Journal Entries & Saved Story Narratives)
CREATE TABLE IF NOT EXISTS album_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    album_id    INTEGER NOT NULL REFERENCES albums(id) ON DELETE CASCADE,
    day_date    TEXT    NOT NULL,           -- e.g. '2026-04-25' or 'General'
    title       TEXT    NOT NULL,           -- e.g. 'Day 1: Hike at Eleven Mile State Park'
    content     TEXT    NOT NULL,           -- note text / narrative
    created_at  TEXT    DEFAULT (datetime('now')),
    updated_at  TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_album_notes_album ON album_notes(album_id);
CREATE INDEX IF NOT EXISTS idx_album_notes_date ON album_notes(day_date);
"""


def init_db() -> None:
    """Create database and tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    """Yield a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_image(conn: sqlite3.Connection, file_path: str, file_hash: str, file_size: int) -> int:
    """Insert or ignore a scanned image. Returns the row id."""
    conn.execute(
        """INSERT INTO images (file_path, file_hash, file_size)
           VALUES (?, ?, ?)
           ON CONFLICT(file_path) DO UPDATE SET
               file_hash = excluded.file_hash,
               file_size = excluded.file_size,
               updated_at = datetime('now')""",
        (file_path, file_hash, file_size),
    )
    row = conn.execute("SELECT id FROM images WHERE file_path = ?", (file_path,)).fetchone()
    return row["id"]


def update_metadata(conn: sqlite3.Connection, image_id: int, **fields) -> None:
    """Update EXIF / geocode metadata fields for an image."""
    allowed = {
        "date_taken", "latitude", "longitude", "camera_make", "camera_model",
        "orientation", "city", "region", "country", "place_name",
    }
    filtered = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not filtered:
        return
    sets = ", ".join(f"{k} = ?" for k in filtered)
    vals = list(filtered.values()) + [image_id]
    conn.execute(
        f"UPDATE images SET {sets}, metadata_done = 1, updated_at = datetime('now') WHERE id = ?",
        vals,
    )


def update_description(conn: sqlite3.Connection, image_id: int, raw: str, enriched: str) -> None:
    """Store VLM description and enriched text."""
    conn.execute(
        """UPDATE images SET raw_description = ?, enriched_text = ?,
           description_done = 1, updated_at = datetime('now') WHERE id = ?""",
        (raw, enriched, image_id),
    )


def mark_embedded(conn: sqlite3.Connection, image_id: int) -> None:
    conn.execute("UPDATE images SET embedded = 1, updated_at = datetime('now') WHERE id = ?", (image_id,))


def get_pending_metadata(conn: sqlite3.Connection, limit: int = 500) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, file_path FROM images WHERE metadata_done = 0 LIMIT ?", (limit,)
    ).fetchall()


def get_pending_descriptions(conn: sqlite3.Connection, limit: int = 500) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, file_path FROM images WHERE description_done = 0 LIMIT ?", (limit,)
    ).fetchall()


def get_pending_embeds(conn: sqlite3.Connection, limit: int = 500) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, enriched_text FROM images WHERE description_done = 1 AND embedded = 0 LIMIT ?",
        (limit,),
    ).fetchall()


def get_image_by_id(conn: sqlite3.Connection, image_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()


def get_stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(metadata_done) as metadata_done,
            SUM(description_done) as described,
            SUM(embedded) as embedded,
            SUM(CASE WHEN date_taken IS NULL OR date_taken = '' THEN 1 ELSE 0 END) as unknown_dates,
            SUM(CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 1 ELSE 0 END) as with_gps
        FROM images
    """).fetchone()
    res = dict(row)
    res["unknown_dates"] = res.get("unknown_dates") or 0
    res["with_gps"] = res.get("with_gps") or 0
    return res



# ── Album Operations ────────────────────────────────────

def create_album(conn: sqlite3.Connection, name: str, description: str = "") -> int:
    """Create a new album with given name and optional description."""
    cursor = conn.execute(
        "INSERT INTO albums (name, description) VALUES (?, ?)",
        (name.strip(), description.strip()),
    )
    return cursor.lastrowid


def get_all_albums(conn: sqlite3.Connection) -> list[dict]:
    """Return all albums with photo count and latest cover image id."""
    rows = conn.execute("""
        SELECT 
            a.id, 
            a.name, 
            a.description, 
            a.created_at,
            COUNT(ai.image_id) as photo_count,
            MAX(ai.image_id) as cover_image_id
        FROM albums a
        LEFT JOIN album_images ai ON a.id = ai.album_id
        GROUP BY a.id
        ORDER BY a.name ASC
    """).fetchall()
    return [dict(r) for r in rows]


def get_album_by_id(conn: sqlite3.Connection, album_id: int) -> dict | None:
    """Fetch an album's details."""
    row = conn.execute("SELECT * FROM albums WHERE id = ?", (album_id,)).fetchone()
    return dict(row) if row else None


def rename_album(conn: sqlite3.Connection, album_id: int, new_name: str, new_description: str = "") -> bool:
    """Rename an album and update its updated_at timestamp."""
    conn.execute(
        "UPDATE albums SET name = ?, description = ?, updated_at = datetime('now') WHERE id = ?",
        (new_name.strip(), new_description.strip(), album_id),
    )
    return True



def add_photo_to_album(conn: sqlite3.Connection, album_id: int, image_id: int) -> bool:
    """Add a photo to an album. Idempotent via OR IGNORE."""
    conn.execute(
        "INSERT OR IGNORE INTO album_images (album_id, image_id) VALUES (?, ?)",
        (album_id, image_id),
    )
    conn.execute(
        "UPDATE albums SET updated_at = datetime('now') WHERE id = ?",
        (album_id,),
    )
    return True


def remove_photo_from_album(conn: sqlite3.Connection, album_id: int, image_id: int) -> bool:
    """Remove a photo from an album."""
    conn.execute(
        "DELETE FROM album_images WHERE album_id = ? AND image_id = ?",
        (album_id, image_id),
    )
    return True


def get_album_image_ids(conn: sqlite3.Connection, album_id: int) -> list[int]:
    """Return image ids belonging to an album."""
    rows = conn.execute(
        "SELECT image_id FROM album_images WHERE album_id = ? ORDER BY added_at DESC",
        (album_id,),
    ).fetchall()
    return [r["image_id"] for r in rows]


def get_photo_albums(conn: sqlite3.Connection, image_id: int) -> list[dict]:
    """Return albums that contain this photo."""
    rows = conn.execute("""
        SELECT a.id, a.name 
        FROM albums a
        JOIN album_images ai ON a.id = ai.album_id
        WHERE ai.image_id = ?
        ORDER BY a.name ASC
    """, (image_id,)).fetchall()
    return [dict(r) for r in rows]


def delete_album(conn: sqlite3.Connection, album_id: int) -> bool:
    """Delete an album (cascade removes associations)."""
    conn.execute("DELETE FROM albums WHERE id = ?", (album_id,))
    return True


def bulk_add_photos_to_album(conn: sqlite3.Connection, album_id: int, image_ids: list[int]) -> int:
    """Batch add multiple images to an album."""
    count = 0
    for iid in image_ids:
        conn.execute(
            "INSERT OR IGNORE INTO album_images (album_id, image_id) VALUES (?, ?)",
            (album_id, iid),
        )
        count += 1
    conn.execute("UPDATE albums SET updated_at = datetime('now') WHERE id = ?", (album_id,))
    return count


def bulk_remove_photos_from_album(conn: sqlite3.Connection, album_id: int, image_ids: list[int]) -> int:
    """Batch remove multiple images from a specific album (images remain in other albums and library)."""
    if not image_ids:
        return 0
    placeholders = ",".join("?" for _ in image_ids)
    cur = conn.execute(
        f"DELETE FROM album_images WHERE album_id = ? AND image_id IN ({placeholders})",
        [album_id] + image_ids,
    )
    conn.execute("UPDATE albums SET updated_at = datetime('now') WHERE id = ?", (album_id,))
    return cur.rowcount


def bulk_move_photos_to_album(conn: sqlite3.Connection, source_album_id: int, target_album_id: int, image_ids: list[int]) -> int:
    """Move multiple images from a source album to a target album."""
    if not image_ids:
        return 0
    # Add to target album
    for iid in image_ids:
        conn.execute(
            "INSERT OR IGNORE INTO album_images (album_id, image_id) VALUES (?, ?)",
            (target_album_id, iid),
        )
    # Remove from source album
    placeholders = ",".join("?" for _ in image_ids)
    conn.execute(
        f"DELETE FROM album_images WHERE album_id = ? AND image_id IN ({placeholders})",
        [source_album_id] + image_ids,
    )
    conn.execute("UPDATE albums SET updated_at = datetime('now') WHERE id = ?", (source_album_id,))
    conn.execute("UPDATE albums SET updated_at = datetime('now') WHERE id = ?", (target_album_id,))
    return len(image_ids)


def delete_images(conn: sqlite3.Connection, image_ids: list[int]) -> list[str]:
    """Delete multiple images from database and return their file paths for disk cleanup."""
    if not image_ids:
        return []
    placeholders = ",".join("?" for _ in image_ids)
    rows = conn.execute(
        f"SELECT file_path FROM images WHERE id IN ({placeholders})", image_ids
    ).fetchall()
    paths = [r["file_path"] for r in rows]

    conn.execute(f"DELETE FROM album_images WHERE image_id IN ({placeholders})", image_ids)
    conn.execute(f"DELETE FROM images WHERE id IN ({placeholders})", image_ids)
    return paths


def bulk_update_location(
    conn: sqlite3.Connection,
    image_ids: list[int],
    place_name: str,
    latitude: float | None,
    longitude: float | None,
    city: str = "",
    region: str = "",
    country: str = "",
) -> list[dict]:
    """Update location metadata for specified images and return updated rows for vector re-embedding."""
    if not image_ids:
        return []
    placeholders = ",".join("?" for _ in image_ids)
    conn.execute(
        f"""UPDATE images 
            SET place_name = ?, latitude = ?, longitude = ?, city = ?, region = ?, country = ?, updated_at = datetime('now')
            WHERE id IN ({placeholders})""",
        [place_name, latitude, longitude, city, region, country] + image_ids,
    )
    rows = conn.execute(
        f"SELECT * FROM images WHERE id IN ({placeholders})", image_ids
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_image_paths(conn: sqlite3.Connection) -> list[tuple[int, str]]:
    """Return all (id, file_path) pairs for thumbnail regeneration."""
    rows = conn.execute("SELECT id, file_path FROM images").fetchall()
    return [(r["id"], r["file_path"]) for r in rows]


# ── Story Cache Operations ──────────────────────────

def get_cached_narrative(conn: sqlite3.Connection, album_id: int, day_date: str) -> dict | None:
    """Fetch a cached narrative for a specific album + day."""
    row = conn.execute(
        "SELECT * FROM story_cache WHERE album_id = ? AND day_date = ?",
        (album_id, day_date),
    ).fetchone()
    return dict(row) if row else None


def upsert_narrative(
    conn: sqlite3.Connection, album_id: int, day_date: str,
    narrative: str, model_used: str, photo_ids: str,
) -> None:
    """Insert or update a cached day narrative."""
    conn.execute(
        """INSERT INTO story_cache (album_id, day_date, narrative, model_used, photo_ids)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(album_id, day_date) DO UPDATE SET
               narrative = excluded.narrative,
               model_used = excluded.model_used,
               photo_ids = excluded.photo_ids,
               created_at = datetime('now')""",
        (album_id, day_date, narrative, model_used, photo_ids),
    )


def clear_story_cache(conn: sqlite3.Connection, album_id: int) -> int:
    """Delete all cached narratives for an album. Returns rows deleted."""
    cur = conn.execute("DELETE FROM story_cache WHERE album_id = ?", (album_id,))
    return cur.rowcount


def delete_day_narrative(conn: sqlite3.Connection, album_id: int, day_date: str) -> int:
    """Delete a single cached day/group narrative for an album. Returns rows deleted."""
    cur = conn.execute(
        "DELETE FROM story_cache WHERE album_id = ? AND day_date = ?",
        (album_id, day_date),
    )
    return cur.rowcount



def get_album_story(conn: sqlite3.Connection, album_id: int) -> list[dict]:
    """Fetch all cached day narratives for an album, ordered chronologically."""
    rows = conn.execute(
        "SELECT * FROM story_cache WHERE album_id = ? ORDER BY day_date ASC",
        (album_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Album Notes Operations ───────────────────────────

def create_album_note(
    conn: sqlite3.Connection,
    album_id: int,
    day_date: str,
    title: str,
    content: str,
) -> dict:
    """Create a new note for an album."""
    cur = conn.execute(
        """INSERT INTO album_notes (album_id, day_date, title, content, created_at, updated_at)
           VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))""",
        (album_id, day_date.strip(), title.strip(), content.strip()),
    )
    note_id = cur.lastrowid
    row = conn.execute(
        """SELECT n.*, a.name AS album_name 
           FROM album_notes n 
           JOIN albums a ON n.album_id = a.id 
           WHERE n.id = ?""",
        (note_id,),
    ).fetchone()
    return dict(row) if row else {}


def save_or_update_story_note(
    conn: sqlite3.Connection,
    album_id: int,
    day_date: str,
    title: str,
    content: str,
) -> dict:
    """
    Save a story day narrative as a note.
    If a note for this album + day already exists, updates it; otherwise creates a new note.
    """
    existing = conn.execute(
        "SELECT id FROM album_notes WHERE album_id = ? AND day_date = ?",
        (album_id, day_date),
    ).fetchone()

    if existing:
        conn.execute(
            """UPDATE album_notes 
               SET title = ?, content = ?, updated_at = datetime('now')
               WHERE id = ?""",
            (title.strip(), content.strip(), existing["id"]),
        )
        note_id = existing["id"]
    else:
        cur = conn.execute(
            """INSERT INTO album_notes (album_id, day_date, title, content, created_at, updated_at)
               VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))""",
            (album_id, day_date.strip(), title.strip(), content.strip()),
        )
        note_id = cur.lastrowid

    row = conn.execute(
        """SELECT n.*, a.name AS album_name 
           FROM album_notes n 
           JOIN albums a ON n.album_id = a.id 
           WHERE n.id = ?""",
        (note_id,),
    ).fetchone()
    return dict(row) if row else {}


def get_album_notes(conn: sqlite3.Connection, album_id: int) -> list[dict]:
    """Retrieve all notes for a specific album, ordered chronologically."""
    rows = conn.execute(
        """SELECT n.*, a.name AS album_name 
           FROM album_notes n 
           JOIN albums a ON n.album_id = a.id 
           WHERE n.album_id = ? 
           ORDER BY n.day_date ASC, n.created_at ASC""",
        (album_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_note_by_id(conn: sqlite3.Connection, note_id: int) -> dict | None:
    """Fetch a single note by ID with album metadata."""
    row = conn.execute(
        """SELECT n.*, a.name AS album_name 
           FROM album_notes n 
           JOIN albums a ON n.album_id = a.id 
           WHERE n.id = ?""",
        (note_id,),
    ).fetchone()
    return dict(row) if row else None


def update_album_note(
    conn: sqlite3.Connection,
    note_id: int,
    title: str,
    content: str,
    day_date: str | None = None,
) -> dict | None:
    """Update title, content, and optional date of an existing note."""
    if day_date:
        conn.execute(
            """UPDATE album_notes 
               SET title = ?, content = ?, day_date = ?, updated_at = datetime('now')
               WHERE id = ?""",
            (title.strip(), content.strip(), day_date.strip(), note_id),
        )
    else:
        conn.execute(
            """UPDATE album_notes 
               SET title = ?, content = ?, updated_at = datetime('now')
               WHERE id = ?""",
            (title.strip(), content.strip(), note_id),
        )
    return get_note_by_id(conn, note_id)


def delete_album_note(conn: sqlite3.Connection, note_id: int) -> bool:
    """Delete a note by ID."""
    cur = conn.execute("DELETE FROM album_notes WHERE id = ?", (note_id,))
    return cur.rowcount > 0


def search_all_notes(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """
    Search notes across all albums by matching title, content, date, or album name.
    """
    q = (query or "").strip().lower()
    if not q:
        rows = conn.execute(
            """SELECT n.*, a.name AS album_name 
               FROM album_notes n 
               JOIN albums a ON n.album_id = a.id 
               ORDER BY n.updated_at DESC 
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    like_pat = f"%{q}%"
    rows = conn.execute(
        """SELECT n.*, a.name AS album_name 
           FROM album_notes n 
           JOIN albums a ON n.album_id = a.id 
           WHERE LOWER(n.title) LIKE ? 
              OR LOWER(n.content) LIKE ? 
              OR LOWER(n.day_date) LIKE ? 
              OR LOWER(a.name) LIKE ? 
           ORDER BY 
              CASE WHEN LOWER(n.title) LIKE ? THEN 0 ELSE 1 END,
              n.updated_at DESC 
           LIMIT ?""",
        (like_pat, like_pat, like_pat, like_pat, like_pat, limit),
    ).fetchall()
    return [dict(r) for r in rows]


