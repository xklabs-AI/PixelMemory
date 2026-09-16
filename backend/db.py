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
    faces_scanned   INTEGER DEFAULT 0,

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

-- Known People and Pets
CREATE TABLE IF NOT EXISTS people (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE COLLATE NOCASE,
    relationship    TEXT DEFAULT 'Friend',  -- 'Spouse', 'Child', 'Parent', 'Sibling', 'Grandparent', 'Friend', 'Pet (Dog)', 'Pet (Cat)', 'Other'
    avatar_face_id  INTEGER,                -- face id for avatar crop
    notes           TEXT DEFAULT '',
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

-- Detected / Tagged Faces in Images
CREATE TABLE IF NOT EXISTS faces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    person_id       INTEGER REFERENCES people(id) ON DELETE SET NULL,
    box_x           REAL NOT NULL,          -- normalized [0.0 - 1.0] coordinates
    box_y           REAL NOT NULL,
    box_w           REAL NOT NULL,
    box_h           REAL NOT NULL,
    confidence      REAL DEFAULT 1.0,       -- detection confidence (1.0 for manual)
    embedding       BLOB,                   -- 128-float binary buffer or null
    is_pet          INTEGER DEFAULT 0,      -- 1 for pet, 0 for human
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_people_name ON people(name);

-- Rejected / False Matches for People and Pets
CREATE TABLE IF NOT EXISTS face_rejections (
    face_id     INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    created_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (face_id, person_id)
);

CREATE INDEX IF NOT EXISTS idx_face_rejections_person ON face_rejections(person_id);
"""


def init_db() -> None:
    """Create database and tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(images)").fetchall()]
        if "faces_scanned" not in cols:
            conn.execute("ALTER TABLE images ADD COLUMN faces_scanned INTEGER DEFAULT 0")
        deduplicate_faces(conn)


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
    if not image_ids:
        return 0
    count = 0
    for iid in image_ids:
        conn.execute(
            "INSERT OR IGNORE INTO album_images (album_id, image_id) VALUES (?, ?)",
            (album_id, iid),
        )
        count += 1
    conn.execute("UPDATE albums SET updated_at = datetime('now') WHERE id = ?", (album_id,))
    return count


def get_albums_for_folder(conn: sqlite3.Connection, folder_path: str) -> list[dict]:
    """Find albums that contain photos from folder_path or whose name matches the folder name."""
    norm_folder = str(Path(folder_path).resolve()).replace("\\", "/")
    folder_name = Path(folder_path).name.strip().lower()

    # 1. Albums containing photos matching folder_path prefix
    rows = conn.execute("""
        SELECT DISTINCT a.id, a.name
        FROM albums a
        JOIN album_images ai ON a.id = ai.album_id
        JOIN images i ON ai.image_id = i.id
        WHERE REPLACE(i.file_path, '\\', '/') LIKE ? || '/%'
           OR REPLACE(i.file_path, '\\', '/') = ?
    """, (norm_folder, norm_folder)).fetchall()

    albums_dict = {r["id"]: dict(r) for r in rows}

    # 2. Match by album name == folder name
    name_rows = conn.execute(
        "SELECT id, name FROM albums WHERE LOWER(name) = ?", (folder_name,)
    ).fetchall()
    for r in name_rows:
        if r["id"] not in albums_dict:
            albums_dict[r["id"]] = dict(r)

    return list(albums_dict.values())


def get_folder_for_album(conn: sqlite3.Connection, album_id: int) -> str | None:
    """Detect the primary folder on disk for an album based on its member photos."""
    rows = conn.execute("""
        SELECT i.file_path
        FROM images i
        JOIN album_images ai ON i.id = ai.image_id
        WHERE ai.album_id = ?
        LIMIT 50
    """, (album_id,)).fetchall()
    if not rows:
        return None
    paths = [Path(r["file_path"]).parent for r in rows if r["file_path"]]
    if not paths:
        return None
    # Find common parent directory
    common = paths[0]
    for p in paths[1:]:
        while common not in p.parents and common != p:
            common = common.parent
    return str(common).replace("\\", "/")


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

    conn.execute(
        f"DELETE FROM face_rejections WHERE face_id IN (SELECT id FROM faces WHERE image_id IN ({placeholders}))",
        image_ids,
    )
    conn.execute(f"DELETE FROM faces WHERE image_id IN ({placeholders})", image_ids)
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


# ── People & Faces ─────────────────────────────────────

def get_faces_for_image(conn: sqlite3.Connection, image_id: int) -> list[dict]:
    """Return all detected / tagged faces for an image."""
    rows = conn.execute(
        """SELECT f.id, f.image_id, f.person_id, f.box_x, f.box_y, f.box_w, f.box_h,
                  f.confidence, f.is_pet, f.created_at,
                  (f.embedding IS NOT NULL) AS has_embedding,
                  p.name AS person_name, p.relationship AS person_relationship
           FROM faces f
           LEFT JOIN people p ON f.person_id = p.id
           WHERE f.image_id = ?
           ORDER BY f.id ASC""",
        (image_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_face_by_id(conn: sqlite3.Connection, face_id: int) -> dict | None:
    """Get single face record with person and image details."""
    row = conn.execute(
        """SELECT f.id, f.image_id, f.person_id, f.box_x, f.box_y, f.box_w, f.box_h,
                  f.confidence, f.is_pet, f.created_at,
                  (f.embedding IS NOT NULL) AS has_embedding,
                  p.name AS person_name, p.relationship AS person_relationship, i.file_path
           FROM faces f
           JOIN images i ON f.image_id = i.id
           LEFT JOIN people p ON f.person_id = p.id
           WHERE f.id = ?""",
        (face_id,),
    ).fetchone()
    return dict(row) if row else None


def insert_face(
    conn: sqlite3.Connection,
    image_id: int,
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
    confidence: float = 1.0,
    embedding: bytes | None = None,
    person_id: int | None = None,
    is_pet: int = 0,
) -> int:
    """Insert a new detected or manual face."""
    cur = conn.execute(
        """INSERT INTO faces (image_id, person_id, box_x, box_y, box_w, box_h, confidence, embedding, is_pet)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (image_id, person_id, box_x, box_y, box_w, box_h, confidence, embedding, is_pet),
    )
    return cur.lastrowid


def update_face_person(conn: sqlite3.Connection, face_id: int, person_id: int | None) -> bool:
    """Assign or unassign a person to a face, ensuring unique tags per image."""
    if person_id is not None:
        row = conn.execute("SELECT image_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        if row:
            img_id = row["image_id"]
            # Enforce unique tags per picture: clear duplicate tags of the same person on this image
            conn.execute(
                "UPDATE faces SET person_id = NULL WHERE image_id = ? AND person_id = ? AND id != ?",
                (img_id, person_id, face_id),
            )
    cur = conn.execute(
        "UPDATE faces SET person_id = ? WHERE id = ?",
        (person_id, face_id),
    )
    return cur.rowcount > 0


def delete_face(conn: sqlite3.Connection, face_id: int) -> bool:
    """Delete a face bounding box and tag."""
    cur = conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
    return cur.rowcount > 0


def get_all_people(conn: sqlite3.Connection) -> list[dict]:
    """Return all known people/pets with face & photo counts."""
    rows = conn.execute(
        """SELECT p.*,
                  COUNT(DISTINCT f.id) AS face_count,
                  COUNT(DISTINCT f.image_id) AS photo_count
           FROM people p
           LEFT JOIN faces f ON f.person_id = p.id
           GROUP BY p.id
           ORDER BY photo_count DESC, p.name ASC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_person_by_id(conn: sqlite3.Connection, person_id: int) -> dict | None:
    """Get person record by ID with stats."""
    row = conn.execute(
        """SELECT p.*,
                  COUNT(DISTINCT f.id) AS face_count,
                  COUNT(DISTINCT f.image_id) AS photo_count
           FROM people p
           LEFT JOIN faces f ON f.person_id = p.id
           WHERE p.id = ?
           GROUP BY p.id""",
        (person_id,),
    ).fetchone()
    return dict(row) if row else None


def get_person_by_name(conn: sqlite3.Connection, name: str) -> dict | None:
    """Find person by case-insensitive name."""
    row = conn.execute(
        "SELECT * FROM people WHERE name = ? COLLATE NOCASE",
        (name.strip(),),
    ).fetchone()
    return dict(row) if row else None


def upsert_person(
    conn: sqlite3.Connection,
    name: str,
    relationship: str = "Friend",
    notes: str = "",
    avatar_face_id: int | None = None,
) -> int:
    """Insert or update a person record by name."""
    existing = get_person_by_name(conn, name)
    if existing:
        conn.execute(
            """UPDATE people 
               SET relationship = COALESCE(NULLIF(?, ''), relationship),
                   notes = COALESCE(NULLIF(?, ''), notes),
                   avatar_face_id = COALESCE(?, avatar_face_id),
                   updated_at = datetime('now')
               WHERE id = ?""",
            (relationship.strip(), notes.strip(), avatar_face_id, existing["id"]),
        )
        return existing["id"]
    else:
        cur = conn.execute(
            """INSERT INTO people (name, relationship, notes, avatar_face_id)
               VALUES (?, ?, ?, ?)""",
            (name.strip(), relationship.strip() or "Friend", notes.strip(), avatar_face_id),
        )
        return cur.lastrowid


def update_person(
    conn: sqlite3.Connection,
    person_id: int,
    name: str | None = None,
    relationship: str | None = None,
    notes: str | None = None,
    avatar_face_id: int | None = None,
) -> dict | None:
    """Update fields on an existing person."""
    fields = []
    vals = []
    if name is not None:
        fields.append("name = ?")
        vals.append(name.strip())
    if relationship is not None:
        fields.append("relationship = ?")
        vals.append(relationship.strip())
    if notes is not None:
        fields.append("notes = ?")
        vals.append(notes.strip())
    if avatar_face_id is not None:
        fields.append("avatar_face_id = ?")
        vals.append(avatar_face_id)

    if not fields:
        return get_person_by_id(conn, person_id)

    fields.append("updated_at = datetime('now')")
    vals.append(person_id)
    conn.execute(f"UPDATE people SET {', '.join(fields)} WHERE id = ?", tuple(vals))
    return get_person_by_id(conn, person_id)


def delete_person(conn: sqlite3.Connection, person_id: int) -> bool:
    """Delete person record and release all tagged faces back to untagged (person_id = NULL)."""
    conn.execute("UPDATE faces SET person_id = NULL WHERE person_id = ?", (person_id,))
    cur = conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
    return cur.rowcount > 0


def get_faces_for_person(conn: sqlite3.Connection, person_id: int) -> list[dict]:
    """Return all tagged face variations / instances for a person across the library."""
    rows = conn.execute(
        """SELECT f.id, f.image_id, f.person_id, f.box_x, f.box_y, f.box_w, f.box_h,
                  f.confidence, f.is_pet, f.created_at,
                  (f.embedding IS NOT NULL) AS has_embedding,
                  i.file_path, i.date_taken, i.place_name,
                  (p.avatar_face_id = f.id) AS is_avatar
           FROM faces f
           JOIN images i ON f.image_id = i.id
           JOIN people p ON f.person_id = p.id
           WHERE f.person_id = ?
           ORDER BY i.date_taken DESC, f.id DESC""",
        (person_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def unlink_face_from_person(conn: sqlite3.Connection, person_id: int, face_id: int) -> bool:
    """Unlink a specific face detection from a person, releasing it back to untagged."""
    cur = conn.execute(
        "UPDATE faces SET person_id = NULL WHERE id = ? AND person_id = ?",
        (face_id, person_id)
    )
    if cur.rowcount == 0:
        return False

    # If this unlinked face was the person's avatar, pick another face or set NULL
    person = get_person_by_id(conn, person_id)
    if person and person.get("avatar_face_id") == face_id:
        next_face = conn.execute(
            "SELECT id FROM faces WHERE person_id = ? LIMIT 1", (person_id,)
        ).fetchone()
        new_avatar = next_face["id"] if next_face else None
        update_person(conn, person_id, avatar_face_id=new_avatar)

    return True


def get_photos_for_person(conn: sqlite3.Connection, person_id: int) -> list[dict]:
    """Return all photos containing this person."""
    rows = conn.execute(
        """SELECT DISTINCT i.*
           FROM images i
           JOIN faces f ON f.image_id = i.id
           WHERE f.person_id = ?
           ORDER BY i.date_taken DESC, i.id DESC""",
        (person_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_people_for_photos(conn: sqlite3.Connection, image_ids: list[int]) -> dict[int, list[dict]]:
    """
    Given a list of image IDs, return a mapping {image_id: [person_dicts]}.
    Used by storyteller to enrich prompts with who is in each photo.
    """
    if not image_ids:
        return {}

    placeholders = ",".join("?" for _ in image_ids)
    rows = conn.execute(
        f"""SELECT f.image_id, f.is_pet, p.id AS person_id, p.name, p.relationship
            FROM faces f
            JOIN people p ON f.person_id = p.id
            WHERE f.image_id IN ({placeholders})
            ORDER BY f.image_id ASC, p.name ASC""",
        image_ids,
    ).fetchall()

    result: dict[int, list[dict]] = {}
    for r in rows:
        img_id = r["image_id"]
        result.setdefault(img_id, []).append({
            "person_id": r["person_id"],
            "name": r["name"],
            "relationship": r["relationship"] or "Friend",
            "is_pet": bool(r["is_pet"]),
        })
    return result


def get_known_face_embeddings(conn: sqlite3.Connection, exclude_face_id: int | None = None) -> list[tuple[int, int, bytes]]:
    """
    Return all tagged faces with embeddings: list of (face_id, person_id, embedding_bytes).
    Used for matching new faces against known people.
    """
    if exclude_face_id:
        rows = conn.execute(
            """SELECT id, person_id, embedding 
               FROM faces 
               WHERE person_id IS NOT NULL AND embedding IS NOT NULL AND id != ?""",
            (exclude_face_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, person_id, embedding 
               FROM faces 
               WHERE person_id IS NOT NULL AND embedding IS NOT NULL"""
        ).fetchall()
    return [(r["id"], r["person_id"], r["embedding"]) for r in rows]


def get_untagged_faces_with_embeddings(conn: sqlite3.Connection, limit: int = 200, exclude_person_id: int | None = None) -> list[dict]:
    """Return untagged faces that have embeddings for batch suggestion/clustering, excluding any rejected for this person."""
    if exclude_person_id:
        rows = conn.execute(
            """SELECT f.id, f.image_id, f.box_x, f.box_y, f.box_w, f.box_h, f.embedding, i.file_path
               FROM faces f
               JOIN images i ON f.image_id = i.id
               WHERE f.person_id IS NULL 
                 AND f.embedding IS NOT NULL
                 AND f.id NOT IN (SELECT face_id FROM face_rejections WHERE person_id = ?)
               LIMIT ?""",
            (exclude_person_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT f.id, f.image_id, f.box_x, f.box_y, f.box_w, f.box_h, f.embedding, i.file_path
               FROM faces f
               JOIN images i ON f.image_id = i.id
               WHERE f.person_id IS NULL AND f.embedding IS NOT NULL
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def reject_face_match(conn: sqlite3.Connection, face_id: int, person_id: int) -> bool:
    """Record that a face is not a match for a person, so it won't be suggested again."""
    conn.execute(
        "INSERT OR IGNORE INTO face_rejections (face_id, person_id) VALUES (?, ?)",
        (face_id, person_id),
    )
    return True


def batch_reject_face_matches(conn: sqlite3.Connection, face_ids: list[int], person_id: int) -> int:
    """Record multiple faces as not matching a person."""
    count = 0
    for fid in face_ids:
        cur = conn.execute(
            "INSERT OR IGNORE INTO face_rejections (face_id, person_id) VALUES (?, ?)",
            (fid, person_id),
        )
        if cur.rowcount > 0:
            count += 1
    return count


def batch_delete_faces(conn: sqlite3.Connection, face_ids: list[int]) -> int:
    """Delete multiple face boxes completely from database."""
    count = 0
    for fid in face_ids:
        if delete_face(conn, fid):
            count += 1
    return count


def calculate_box_iou(box1: tuple, box2: tuple) -> float:
    """Compute Intersection over Union between two (x, y, w, h) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2]) * max(0.0, box1[3])
    area2 = max(0.0, box2[2]) * max(0.0, box2[3])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_faces(conn: sqlite3.Connection) -> int:
    """Find and merge/remove duplicate face bounding boxes on the same image."""
    rows = conn.execute(
        """SELECT id, image_id, person_id, box_x, box_y, box_w, box_h, confidence,
                  (embedding IS NOT NULL) AS has_emb, is_pet, created_at
           FROM faces
           ORDER BY image_id, (person_id IS NOT NULL) DESC, (embedding IS NOT NULL) DESC, id ASC"""
    ).fetchall()

    by_image: dict[int, list[sqlite3.Row]] = {}
    for r in rows:
        by_image.setdefault(r["image_id"], []).append(r)

    deleted_count = 0
    for image_id, face_list in by_image.items():
        kept: list[dict] = []
        for row in face_list:
            f = dict(row)
            box_f = (f["box_x"], f["box_y"], f["box_w"], f["box_h"])
            duplicate_of = None
            for k in kept:
                box_k = (k["box_x"], k["box_y"], k["box_w"], k["box_h"])
                iou = calculate_box_iou(box_f, box_k)
                center_dist = abs(f["box_x"] - k["box_x"]) + abs(f["box_y"] - k["box_y"])
                size_diff = abs(f["box_w"] - k["box_w"]) + abs(f["box_h"] - k["box_h"])
                if iou > 0.55 or (center_dist < 0.08 and size_diff < 0.12):
                    duplicate_of = k
                    break

            if duplicate_of is not None:
                if not duplicate_of.get("person_id") and f.get("person_id"):
                    conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (f["person_id"], duplicate_of["id"]))
                    duplicate_of["person_id"] = f["person_id"]
                conn.execute("DELETE FROM face_rejections WHERE face_id = ?", (f["id"],))
                conn.execute("DELETE FROM faces WHERE id = ?", (f["id"],))
                deleted_count += 1
            else:
                kept.append(f)

    if deleted_count > 0:
        conn.commit()

    # Phase 2: Enforce unique tags per picture (a single photo cannot have duplicate tags for the same person/pet)
    tagged_rows = conn.execute(
        """SELECT id, image_id, person_id, box_x, box_y, box_w, box_h, confidence,
                  (embedding IS NOT NULL) AS has_emb, is_pet
           FROM faces
           WHERE person_id IS NOT NULL
           ORDER BY image_id, person_id, has_emb DESC, confidence DESC, id ASC"""
    ).fetchall()

    by_tag: dict[tuple[int, int], list[sqlite3.Row]] = {}
    for r in tagged_rows:
        by_tag.setdefault((r["image_id"], r["person_id"]), []).append(r)

    tag_dups_deleted = 0
    for (img_id, pid), flist in by_tag.items():
        if len(flist) > 1:
            # Keep the primary face box (highest confidence/embedding)
            primary_face_id = flist[0]["id"]
            # Ensure person's avatar points to a valid face
            conn.execute(
                "UPDATE people SET avatar_face_id = ? WHERE id = ? AND (avatar_face_id IS NULL OR avatar_face_id NOT IN (SELECT id FROM faces))",
                (primary_face_id, pid),
            )
            # Remove the duplicate tagged detection boxes
            for dup in flist[1:]:
                conn.execute("DELETE FROM face_rejections WHERE face_id = ?", (dup["id"],))
                conn.execute("DELETE FROM faces WHERE id = ?", (dup["id"],))
                tag_dups_deleted += 1

    if tag_dups_deleted > 0:
        conn.commit()

    return deleted_count + tag_dups_deleted




