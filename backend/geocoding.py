"""
Hybrid Global Location Search Engine.
Combines:
1. Live OpenStreetMap Photon / Nominatim API for global search (cities, towns, parks, landmarks, lakes, mountains, etc.)
2. Persistent local SQLite cache for instant response (<1ms) and offline capability
3. Offline reverse_geocoder + supplementary dataset fallback when network is unavailable
4. Multi-token scoring with US state abbreviation expansion
"""

import json
import logging
import sqlite3
import time
import urllib.parse
import urllib.request
from typing import Optional
from pathlib import Path

from backend.config import DATA_DIR
from backend.places import US_STATES, SUPPLEMENTARY_PLACES

logger = logging.getLogger("pixelmemory.geocoding")

# Dedicated location cache database
CACHE_DB_PATH = DATA_DIR / "location_cache.db"


def get_cache_db() -> sqlite3.Connection:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS location_queries (
            query_key TEXT PRIMARY KEY,
            results_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_query_key ON location_queries(query_key)")
    conn.commit()
    return conn


def get_cached_locations(query_key: str) -> Optional[list[dict]]:
    try:
        with get_cache_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT results_json FROM location_queries WHERE query_key = ?", (query_key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
    except Exception as e:
        logger.debug(f"Cache read error: {e}")
    return None


def cache_locations(query_key: str, results: list[dict]):
    try:
        with get_cache_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO location_queries (query_key, results_json, created_at) VALUES (?, ?, ?)",
                (query_key, json.dumps(results), time.time()),
            )
            conn.commit()
    except Exception as e:
        logger.debug(f"Cache write error: {e}")


def format_osm_place(feature: dict) -> Optional[dict]:
    """Convert an OpenStreetMap Photon feature to standard place dictionary."""
    props = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    coords = geometry.get("coordinates", [])
    if len(coords) < 2:
        return None

    lon, lat = float(coords[0]), float(coords[1])
    name = props.get("name") or props.get("city") or props.get("district") or props.get("state")
    if not name:
        return None

    city = props.get("city") or props.get("town") or props.get("village") or props.get("hamlet") or ""
    county = props.get("county") or ""
    state = props.get("state") or props.get("region") or ""
    country = props.get("country") or props.get("countrycode", "").upper()
    country_code = (props.get("countrycode") or "").upper()

    parts = []
    parts.append(name)
    if city and city.lower() != name.lower():
        parts.append(city)
    if state and state.lower() != name.lower():
        parts.append(state)
    if country and country.lower() not in (name.lower(), state.lower()):
        parts.append(country)

    place_name = ", ".join(parts)

    return {
        "name": name,
        "city": city or name,
        "region": state,
        "county": county,
        "country": country_code or country,
        "place_name": place_name,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "osm_type": props.get("osm_value") or props.get("type", "place"),
    }


def search_osm_photon(query: str, limit: int = 15) -> list[dict]:
    """Search live OpenStreetMap via Photon autocomplete API."""
    q_encoded = urllib.parse.quote(query.strip())
    url = f"https://photon.komoot.io/api/?q={q_encoded}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "PixelMemory-PhotoSearch/1.0"})

    with urllib.request.urlopen(req, timeout=2.5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        features = data.get("features", [])
        places = []
        seen = set()
        for feat in features:
            p = format_osm_place(feat)
            if p:
                key = (round(p["latitude"], 3), round(p["longitude"], 3))
                if key not in seen:
                    seen.add(key)
                    places.append(p)
        return places


_rg_singleton = None


def get_offline_geocoder():
    global _rg_singleton
    if _rg_singleton is None:
        import reverse_geocoder as rg
        _rg_singleton = rg.RGeocoder(mode=1)
    return _rg_singleton


def search_offline_fallback(query: str, limit: int = 12) -> list[dict]:
    """Smart offline search across reverse_geocoder database and supplementary landmarks."""
    raw_q = (query or "").strip().lower()
    if not raw_q or len(raw_q) < 2:
        return []

    comma_parts = [p.strip() for p in raw_q.split(",") if p.strip()]
    primary_query = comma_parts[0] if comma_parts else raw_q
    all_tokens = [t for t in raw_q.replace(",", " ").split() if t]

    state_hints = set()
    for token in all_tokens + comma_parts:
        if token in US_STATES:
            state_hints.add(US_STATES[token].lower())
            state_hints.add(token)

    candidates = []

    def score_location(name: str, region: str, county: str, country: str, pop: int = 0, is_supplementary: bool = False) -> int:
        name_lower = name.lower()
        region_lower = (region or "").lower()
        country_lower = (country or "").lower()
        county_lower = (county or "").lower()

        score = 0
        if name_lower == raw_q:
            score += 1200
        elif raw_q in name_lower:
            score += 400
        elif name_lower.startswith(raw_q):
            score += 600

        if len(comma_parts) > 1:
            if name_lower == primary_query:
                score += 1000
            elif name_lower.startswith(primary_query):
                score += 500
            elif primary_query in name_lower:
                score += 250

        words = name_lower.split()
        for w in words:
            if w.startswith(primary_query):
                score += 200
                break

        if len(comma_parts) > 1:
            sec = comma_parts[1].strip()
            if sec in US_STATES and US_STATES[sec].lower() == region_lower:
                score += 800
            elif sec == region_lower or sec == country_lower:
                score += 700
            elif sec in region_lower:
                score += 400
        elif state_hints:
            if region_lower in state_hints:
                score += 600

        matched_tokens = 0
        full_text = f"{name_lower} {region_lower} {county_lower} {country_lower}"
        for token in all_tokens:
            if token in full_text:
                matched_tokens += 1
                if token in name_lower:
                    matched_tokens += 1

        if matched_tokens >= len(all_tokens):
            score += 300

        if pop > 0:
            score += min(250, int(pop / 3000))
        if is_supplementary:
            score += 100

        return score

    # Supplementary places
    for p in SUPPLEMENTARY_PLACES:
        name = p["name"]
        region = p.get("region", "")
        county = p.get("county", "")
        country = p.get("country", "US")
        pop = p.get("pop", 10000)

        s = score_location(name, region, county, country, pop=pop, is_supplementary=True)
        if s > 150:
            parts = [x for x in (name, region, country) if x]
            candidates.append({
                "name": name,
                "city": name,
                "region": region,
                "county": county,
                "country": country,
                "place_name": ", ".join(parts),
                "latitude": float(p["lat"]),
                "longitude": float(p["lon"]),
                "score": s,
            })

    # Global offline DB
    geo = get_offline_geocoder()
    for loc in geo.locations:
        name = loc.get("name", "")
        region = loc.get("admin1", "")
        county = loc.get("admin2", "")
        country = loc.get("cc", "")

        name_lower = name.lower()
        if not (primary_query in name_lower or any(t in name_lower for t in all_tokens if len(t) >= 3)):
            continue

        s = score_location(name, region, county, country, pop=0, is_supplementary=False)
        if s > 150:
            parts = [x for x in (name, region, country) if x]
            candidates.append({
                "name": name,
                "city": name,
                "region": region,
                "county": county,
                "country": country,
                "place_name": ", ".join(parts),
                "latitude": float(loc["lat"]),
                "longitude": float(loc["lon"]),
                "score": s,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    results = []
    for item in candidates:
        key = (item["name"].lower(), item["region"].lower(), item["country"].lower())
        if key not in seen:
            seen.add(key)
            item_clean = {k: v for k, v in item.items() if k != "score"}
            results.append(item_clean)
            if len(results) >= limit:
                break
    return results


def search_locations_hybrid(query: str, limit: int = 15) -> list[dict]:
    """
    Main robust entrypoint for location search:
    1. Check local persistent cache
    2. Try live OpenStreetMap Photon API
    3. Seamlessly fallback to offline dataset if offline or on network error
    """
    q_norm = (query or "").strip().lower()
    if not q_norm or len(q_norm) < 2:
        return []

    cache_key = f"q:{q_norm}:lim:{limit}"
    cached = get_cached_locations(cache_key)
    if cached:
        return cached

    results = []
    try:
        # Live OSM search with 2.5s timeout
        results = search_osm_photon(query, limit=limit)
    except Exception as e:
        logger.warning(f"Online geocoding unavailable or timed out: {e}, using offline dataset.")

    # If OSM returned no results (or went offline), fallback to offline search
    if not results:
        results = search_offline_fallback(query, limit=limit)

    if results:
        cache_locations(cache_key, results)

    return results
