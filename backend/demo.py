"""Generate and index a curated demo photo archive for immediate exploration."""

import os
import math
import random
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from backend.config import DATA_DIR, THUMB_DIR, THUMB_SIZE
from backend.db import init_db, get_conn, upsert_image, update_metadata, update_description, mark_embedded
from backend.scanner import hash_file
from backend.search import SemanticSearch

DEMO_DIR = DATA_DIR / "demo_photos"

# Curated high-fidelity sample memories
DEMO_ITEMS = [
    {
        "filename": "grand_canyon_sunset.jpg",
        "date_taken": "2016-03-14T17:42:10",
        "latitude": 36.0544,
        "longitude": -112.1401,
        "city": "Grand Canyon Village",
        "region": "Arizona",
        "country": "US",
        "place_name": "Grand Canyon National Park, Arizona, US",
        "camera_make": "Canon",
        "camera_model": "EOS 5D Mark III",
        "raw_description": (
            "A breathtaking golden hour view overlooking the deep red rock ravines of the Grand Canyon. "
            "Three hikers stand at a rocky viewpoint admiring the dramatic orange and purple sunset sky. "
            "Warm evening lighting casts long shadows across the rugged plateau cliffs."
        ),
        "theme": ("#2b1055", "#75225b", "#d85741", "#f7b05b"),
        "title": "Grand Canyon Sunset",
        "icon": "⛰️",
    },
    {
        "filename": "birthday_party_celebration.jpg",
        "date_taken": "2021-08-22T19:15:00",
        "latitude": 47.6062,
        "longitude": -122.3321,
        "city": "Seattle",
        "region": "Washington",
        "country": "US",
        "place_name": "Seattle, Washington, US",
        "camera_make": "Apple",
        "camera_model": "iPhone 12 Pro",
        "raw_description": (
            "A cheerful indoor birthday party around a wooden dining table. A smiling young woman blows out "
            "candles on a multi-tiered chocolate berry cake surrounded by five laughing friends wearing party hats. "
            "Fairy lights and colorful balloons decorate the cozy living room in the background."
        ),
        "theme": ("#1c1917", "#44283c", "#831843", "#f43f5e"),
        "title": "Birthday Celebration",
        "icon": "🎂",
    },
    {
        "filename": "snow_day_dog_park.jpg",
        "date_taken": "2022-01-18T11:30:25",
        "latitude": 39.1911,
        "longitude": -106.8175,
        "city": "Aspen",
        "region": "Colorado",
        "country": "US",
        "place_name": "Aspen, Colorado, US",
        "camera_make": "Sony",
        "camera_model": "ILCE-7M4",
        "raw_description": (
            "An energetic golden retriever happily leaping through fresh deep powder snow catching a bright red frisbee. "
            "Pine trees heavy with white snow and towering snow-capped mountain peaks line the winter horizon under crisp blue skies."
        ),
        "theme": ("#0f172a", "#1e3a8a", "#38bdf8", "#f0f9ff"),
        "title": "Snow Day with Dog",
        "icon": "🐕",
    },
    {
        "filename": "tokyo_shibuya_night_market.jpg",
        "date_taken": "2019-10-05T21:40:12",
        "latitude": 35.6595,
        "longitude": 139.7005,
        "city": "Shibuya",
        "region": "Tokyo",
        "country": "JP",
        "place_name": "Shibuya, Tokyo, JP",
        "camera_make": "Fujifilm",
        "camera_model": "X100V",
        "raw_description": (
            "Vibrant night scene in an illuminated alleyway packed with ramen food stalls and glowing neon lanterns in Japanese kanji. "
            "Pedestrians with umbrellas stroll on wet pavement reflecting red, magenta, and cyan neon lights."
        ),
        "theme": ("#09090b", "#4c0519", "#9f1239", "#38bdf8"),
        "title": "Tokyo Night Market",
        "icon": "🏮",
    },
    {
        "filename": "honolulu_beach_palms.jpg",
        "date_taken": "2023-06-12T16:05:40",
        "latitude": 21.2766,
        "longitude": -157.8273,
        "city": "Honolulu",
        "region": "Hawaii",
        "country": "US",
        "place_name": "Waikiki Beach, Honolulu, Hawaii, US",
        "camera_make": "Apple",
        "camera_model": "iPhone 14 Pro",
        "raw_description": (
            "A serene tropical beach scene with turquoise ocean waves gently lapping soft white sand. "
            "Tall curved palm trees frame the shoreline with surfers in the distance under a sunny afternoon sky."
        ),
        "theme": ("#042f2e", "#0f766e", "#14b8a6", "#fef08a"),
        "title": "Waikiki Tropical Beach",
        "icon": "🌴",
    },
    {
        "filename": "swiss_alps_hiking_matterhorn.jpg",
        "date_taken": "2018-07-29T10:15:33",
        "latitude": 45.9763,
        "longitude": 7.7491,
        "city": "Zermatt",
        "region": "Valais",
        "country": "CH",
        "place_name": "Zermatt, Valais, CH",
        "camera_make": "Sony",
        "camera_model": "ILCE-7RM3",
        "raw_description": (
            "Two hikers with heavy backpacks walking a winding dirt alpine trail toward a crystal clear glacial mountain lake. "
            "The iconic jagged peak of the Matterhorn rises sharply into crisp mountain air with wildflower meadows in the foreground."
        ),
        "theme": ("#064e3b", "#047857", "#10b981", "#67e8f9"),
        "title": "Swiss Alps Hiking",
        "icon": "🏔️",
    },
    {
        "filename": "cozy_coffee_shop_reading.jpg",
        "date_taken": "2020-11-03T14:22:15",
        "latitude": 45.5152,
        "longitude": -122.6784,
        "city": "Portland",
        "region": "Oregon",
        "country": "US",
        "place_name": "Portland, Oregon, US",
        "camera_make": "Leica",
        "camera_model": "Q2",
        "raw_description": (
            "Warm aesthetic close-up of a ceramic cup of hot latte with intricate fern leaf foam art resting on a rustic wooden table. "
            "Beside it sits an open hardcover book and vintage spectacles in soft morning window light against exposed brick."
        ),
        "theme": ("#291508", "#593318", "#a26739", "#f5d0a6"),
        "title": "Cozy Coffee & Book",
        "icon": "☕",
    },
    {
        "filename": "big_sur_vintage_car_drive.jpg",
        "date_taken": "2017-09-19T18:10:50",
        "latitude": 36.3168,
        "longitude": -121.8885,
        "city": "Big Sur",
        "region": "California",
        "country": "US",
        "place_name": "Big Sur, California, US",
        "camera_make": "Nikon",
        "camera_model": "D850",
        "raw_description": (
            "A classic vintage cherry red convertible automobile driving across the Bixby Creek Bridge on Highway 1. "
            "Majestic Pacific coastal cliffs, breaking white ocean foam, and golden seaside sunset fog envelope the scenic route."
        ),
        "theme": ("#18181b", "#881337", "#e11d48", "#fdba74"),
        "title": "Big Sur Pacific Highway",
        "icon": "🚗",
    },
    {
        "filename": "desert_milky_way_stargazing.jpg",
        "date_taken": "2021-05-15T23:55:18",
        "latitude": 38.7331,
        "longitude": -109.5925,
        "city": "Moab",
        "region": "Utah",
        "country": "US",
        "place_name": "Arches National Park, Moab, Utah, US",
        "camera_make": "Sony",
        "camera_model": "ILCE-7S3",
        "raw_description": (
            "Stunning long exposure astrophotography of the glowing purple and silver Milky Way galaxy arched over Delicate Arch. "
            "A silhouette of two stargazers sits on red sandstone rocks under thousands of brilliant stars in an ink black night."
        ),
        "theme": ("#020617", "#1e1b4b", "#4338ca", "#a855f7"),
        "title": "Milky Way over Desert",
        "icon": "🌌",
    },
    {
        "filename": "central_park_family_picnic.jpg",
        "date_taken": "2015-05-24T13:45:00",
        "latitude": 40.7851,
        "longitude": -73.9683,
        "city": "New York",
        "region": "New York",
        "country": "US",
        "place_name": "Central Park, New York, NY, US",
        "camera_make": "Canon",
        "camera_model": "EOS 6D",
        "raw_description": (
            "A happy family of four sitting on a red and white checkered picnic blanket on the Sheep Meadow lawn. "
            "Children are drinking lemonade and eating sandwiches with lush green spring elm trees and Manhattan skyline in the backdrop."
        ),
        "theme": ("#052e16", "#15803d", "#4ade80", "#fef08a"),
        "title": "Central Park Picnic",
        "icon": "🧺",
    },
    {
        "filename": "kyoto_autumn_maple_temple.jpg",
        "date_taken": "2022-11-17T15:20:44",
        "latitude": 34.9949,
        "longitude": 135.7850,
        "city": "Kyoto",
        "region": "Kyoto",
        "country": "JP",
        "place_name": "Kiyomizu-dera, Kyoto, JP",
        "camera_make": "Sony",
        "camera_model": "ILCE-7C",
        "raw_description": (
            "Spectacular autumn foliage ablaze with scarlet red and golden Japanese maple leaves framing a traditional wooden pagoda. "
            "A peaceful stone pathway and tranquil zen garden with raked sand creates a serene contemplative mood."
        ),
        "theme": ("#1c1917", "#7f1d1d", "#dc2626", "#fbbf24"),
        "title": "Kyoto Autumn Leaves",
        "icon": "🍁",
    },
    {
        "filename": "chicago_architecture_dusk.jpg",
        "date_taken": "2023-09-08T19:35:10",
        "latitude": 41.8818,
        "longitude": -87.6231,
        "city": "Chicago",
        "region": "Illinois",
        "country": "US",
        "place_name": "Downtown, Chicago, Illinois, US",
        "camera_make": "Panasonic",
        "camera_model": "DC-S5M2",
        "raw_description": (
            "Modern glass skyscrapers reflected in the calm waters of the Chicago River during blue hour. "
            "Architectural bridges cross the canal with illuminated city tour boats gliding beneath glittering high-rise office windows."
        ),
        "theme": ("#030712", "#1e293b", "#3b82f6", "#f59e0b"),
        "title": "Chicago River Skyline",
        "icon": "🏙️",
    },
]


def _hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def _generate_artistic_fixture(filepath: Path, item: dict, size: tuple[int, int] = (800, 800)) -> None:
    """Renders a visually pleasing aesthetic gradient and illustration fixture."""
    w, h = size
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    colors = [_hex_to_rgb(c) for c in item["theme"]]

    # Vertical multi-stop gradient
    for y in range(h):
        t = y / (h - 1)
        # Find which color segment we are in
        segment_count = len(colors) - 1
        seg_index = min(int(t * segment_count), segment_count - 1)
        sub_t = (t - seg_index / segment_count) * segment_count

        c1 = colors[seg_index]
        c2 = colors[seg_index + 1]
        r = int(c1[0] + (c2[0] - c1[0]) * sub_t)
        g = int(c1[1] + (c2[1] - c1[1]) * sub_t)
        b = int(c1[2] + (c2[2] - c1[2]) * sub_t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Artistic horizon / geometric contours
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ol_draw = ImageDraw.Draw(overlay)

    # Ambient glow circle
    cx, cy = w // 2, int(h * 0.45)
    radius = int(w * 0.35)
    for r_step in range(radius, 0, -8):
        alpha = int(18 * (1 - r_step / radius))
        glow_col = (*colors[-1], alpha)
        ol_draw.ellipse([cx - r_step, cy - r_step, cx + r_step, cy + r_step], fill=glow_col)

    # Mountain / curve silhouettes at bottom
    points = [(0, h)]
    for x in range(0, w + 1, 15):
        curve_y = int(h * 0.65 + math.sin(x * 0.012) * 45 + math.cos(x * 0.025) * 25)
        points.append((x, curve_y))
    points.append((w, h))
    ol_draw.polygon(points, fill=(colors[0][0], colors[0][1], colors[0][2], 210))

    # Secondary foreground contour
    fg_points = [(0, h)]
    for x in range(0, w + 1, 15):
        curve_y = int(h * 0.78 + math.cos(x * 0.015) * 35 + math.sin(x * 0.03) * 15)
        fg_points.append((x, curve_y))
    fg_points.append((w, h))
    ol_draw.polygon(fg_points, fill=(10, 10, 15, 235))

    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))

    # Watermark text
    try:
        font_large = ImageFont.load_default()
    except Exception:
        font_large = None

    draw = ImageDraw.Draw(img)
    # Subtle metadata text watermark
    dt = datetime.fromisoformat(item["date_taken"]).strftime("%B %Y")
    label = f"{item['title']} · {dt}"
    draw.rectangle([(24, h - 68), (w - 24, h - 24)], fill=(0, 0, 0, 140))
    draw.text((36, h - 54), label, fill=(240, 240, 240), font=font_large)
    draw.text((36, h - 38), f"📍 {item['place_name']} · 📷 {item['camera_model']}", fill=(180, 180, 190), font=font_large)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    img.save(filepath, "JPEG", quality=88)


def seed_demo_archive(clear_existing: bool = False) -> dict:
    """Generates sample photos, populates SQLite, and embeds in ChromaDB."""
    init_db()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    search = SemanticSearch()

    if clear_existing:
        with get_conn() as conn:
            conn.execute("DELETE FROM images")

    seeded = 0
    batch_embed_items = []

    with get_conn() as conn:
        for idx, item in enumerate(DEMO_ITEMS, start=1):
            fpath = DEMO_DIR / item["filename"]

            # 1. Generate image fixture if not already present
            if not fpath.exists():
                _generate_artistic_fixture(fpath, item)

            fhash = hash_file(str(fpath))
            fsize = os.path.getsize(fpath)

            # 2. Upsert into database
            image_id = upsert_image(conn, file_path=str(fpath), file_hash=fhash, file_size=fsize)

            # 3. Create thumbnail
            thumb_path = THUMB_DIR / f"{image_id}.jpg"
            if not thumb_path.exists():
                with Image.open(fpath) as im:
                    im.thumbnail(THUMB_SIZE)
                    if im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")
                    im.save(thumb_path, "JPEG", quality=80)

            # 4. Update metadata
            update_metadata(
                conn,
                image_id=image_id,
                date_taken=item["date_taken"],
                latitude=item["latitude"],
                longitude=item["longitude"],
                city=item["city"],
                region=item["region"],
                country=item["country"],
                place_name=item["place_name"],
                camera_make=item["camera_make"],
                camera_model=item["camera_model"],
            )

            # 5. Format enriched text
            dt = datetime.fromisoformat(item["date_taken"]).strftime("%B %Y")
            enriched = f"{dt}, {item['place_name']}, taken with {item['camera_model']} — {item['raw_description']}"
            update_description(conn, image_id, raw=item["raw_description"], enriched=enriched)

            batch_embed_items.append((image_id, enriched))
            seeded += 1

    # 6. Batch embed all enriched descriptions into ChromaDB
    if batch_embed_items:
        search.add_batch(batch_embed_items)
        with get_conn() as conn:
            for image_id, _ in batch_embed_items:
                mark_embedded(conn, image_id)

    return {
        "seeded": seeded,
        "total_vectors": search.count,
        "demo_dir": str(DEMO_DIR),
    }


if __name__ == "__main__":
    res = seed_demo_archive()
    print(f"Successfully seeded {res['seeded']} demo photos! Total vectors: {res['total_vectors']}")
