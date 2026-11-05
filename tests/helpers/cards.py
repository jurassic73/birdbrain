"""A complete species info card (and its sounds) as species_info.build() would return it, for cache and page tests."""

CARD = {
    "name": "Bushtit", "query": "Bushtit", "common": "Bushtit", "sci_name": "Psaltriparus minimus", "inat_id": 14886,
    "is_bird": True, "photo": {"url": "https://example.org/p.jpg", "credit": "(c) Someone, some rights reserved (CC BY)",
                               "license": "CC BY"},
    "description": "A tiny gray songbird.", "notes": ["Eats insects.", "Travels in chattering flocks."],
    "local_counts": [5, 5, 6, 8, 9, 7, 6, 5, 4, 5, 6, 5], "status": "Year-round resident",
    "expected_months": list(range(1, 13)), "place_name": "Clark County, US, WA", "map_center": [45.7, -122.5],
    "links": [{"label": "eBird", "url": "https://ebird.org/species/bushti"}],
}

SOUNDS = [{"label": "Song", "file": "https://xeno-canto.org/1/download", "id": 1},
          {"label": "Call", "file": "https://xeno-canto.org/2/download", "id": 2}]
