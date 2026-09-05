"""Lightweight local common synonym mapping for visual photo concepts."""

# Curated bidirectional/cluster mappings for common visual photo domains.
# Kept simple, concise, and focused on terms commonly used by VLMs vs user queries.
COMMON_VISUAL_SYNONYMS: dict[str, list[str]] = {
    # Architecture, structures, and condition
    "building": ["structure", "architecture", "house", "ruins", "castle", "tower", "church", "edifice", "monument"],
    "structure": ["building", "architecture", "ruins", "monument"],
    "derelict": ["abandoned", "ruins", "ruined", "ancient", "dilapidated", "crumbling", "decayed"],
    "abandoned": ["derelict", "ruins", "ruined", "empty", "deserted"],
    "ruins": ["derelict", "abandoned", "ancient", "historical", "monument", "stone"],
    "ancient": ["ruins", "historical", "antique", "old"],

    # Nature, landscape, and environment (distinct water bodies without generic hypernym 'water')
    "bush": ["shrub", "foliage", "hedge", "plant", "greenery"],
    "shrub": ["bush", "foliage", "hedge", "greenery"],
    "foliage": ["leaves", "greenery", "bush", "trees", "vegetation"],
    "beach": ["shore", "coast", "seaside", "coastline", "oceanfront"],
    "ocean": ["sea", "seaside", "marine", "saltwater"],
    "sea": ["ocean", "marine", "saltwater", "seaside"],
    "lake": ["pond", "reservoir", "lagoon"],
    "pond": ["lake", "pool", "lagoon"],
    "river": ["stream", "creek", "waterway", "brook"],
    "stream": ["river", "creek", "waterway", "brook"],
    "forest": ["woods", "trees", "woodland"],

    # Food & culinary categories (strict: pasta/noodles distinct from pizza)
    "pasta": ["noodles", "spaghetti", "macaroni", "penne", "linguine"],
    "noodles": ["pasta", "spaghetti", "ramen", "chow mein"],
    "pizza": ["flatbread"],

    # People
    "human": ["person", "man", "woman", "child", "people", "individual"],
    "person": ["human", "man", "woman", "child", "people", "individual"],
    "man": ["person", "human", "guy", "male"],
    "woman": ["person", "human", "lady", "female"],
    "child": ["kid", "boy", "girl", "toddler", "baby"],
    "people": ["humans", "persons", "crowd", "group"],

    # Animals
    "dog": ["puppy", "canine", "hound", "pup"],
    "puppy": ["dog", "pup", "canine"],
    "cat": ["kitten", "feline", "kitty"],
    "kitten": ["cat", "kitty", "feline"],
    "rabbit": ["bunny", "hare"],
    "bird": ["avian", "fowl"],

    # Vehicles & Transportation
    "car": ["automobile", "vehicle", "sedan", "sports car", "suv", "motorcar"],
    "automobile": ["car", "vehicle", "sedan"],
    "vehicle": ["car", "automobile", "truck"],
    "airplane": ["plane", "aircraft", "aeroplane", "jet"],
    "plane": ["airplane", "aircraft", "jet"],

    # Objects & Apparel
    "cone": ["cone collar", "collar", "protective cone", "e-collar"],
    "mug": ["cup", "coffee mug", "coffee cup"],
    "cup": ["mug", "glass"],
    "umbrella": ["parasol"],

    # Atmosphere, light & weather
    "sunset": ["sundown", "dusk", "twilight", "golden hour"],
    "sunrise": ["dawn", "sunup", "daybreak"],
}


def get_entity_synonyms(word: str) -> list[str]:
    """Retrieve synonyms for a given word or root token."""
    w = word.strip().lower()
    if not w:
        return []
    syns = COMMON_VISUAL_SYNONYMS.get(w, [])
    if not syns and w.endswith("s") and len(w) > 3:
        # Singular fallback
        syns = COMMON_VISUAL_SYNONYMS.get(w[:-1], [])
    return syns


def expand_query_text(query: str) -> str:
    """
    Generate an expanded semantic query string containing primary synonyms
    to boost initial ANN candidate recall in vector databases.
    """
    words = [w.strip(".,;:?!'\"()[]{}").lower() for w in query.split()]
    added = []
    for w in words:
        syns = get_entity_synonyms(w)
        if syns:
            added.extend(syns[:2])
    if added:
        return query + " " + " ".join(dict.fromkeys(added))
    return query
