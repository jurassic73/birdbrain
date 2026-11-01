from datetime import datetime, timedelta

import httpx

import pytest
from fastapi.testclient import TestClient

from birdbrain import db, species_info
from birdbrain.web import app as webapp

CARD = {
    "name": "Bushtit", "query": "Bushtit", "common": "Bushtit", "sci_name": "Psaltriparus minimus", "inat_id": 14886,
    "is_bird": True, "photo": {"url": "https://example.org/p.jpg", "credit": "(c) Someone, some rights reserved (CC BY)",
                               "license": "CC BY"},
    "description": "A tiny gray songbird.", "notes": ["Eats insects.", "Travels in chattering flocks."],
    "local_counts": [5, 5, 6, 8, 9, 7, 6, 5, 4, 5, 6, 5], "status": "Year-round resident",
    "expected_months": list(range(1, 13)), "place_name": "Clark County, US, WA", "map_center": [45.7, -122.5],
    "links": [{"label": "eBird", "url": "https://ebird.org/species/bushti"}],
}


@pytest.fixture
def fake_build(monkeypatch):
    calls = []

    def build(name):
        calls.append(name)
        return {**CARD, "name": name}
    monkeypatch.setattr(species_info, "build", build)
    return calls


def test_local_status_classes():
    assert species_info.local_status([10] * 12) == ("Year-round resident", list(range(1, 13)))
    summer = [0, 0, 0, 20, 40, 50, 50, 40, 20, 0, 0, 0]
    assert species_info.local_status(summer)[0] == "Summer visitor"
    winter = [50, 40, 20, 2, 0, 0, 0, 0, 0, 20, 40, 50]
    assert species_info.local_status(winter)[0] == "Winter visitor"
    passage = [0, 0, 0, 30, 40, 2, 0, 1, 30, 40, 0, 0]
    assert species_info.local_status(passage)[0] == "Passes through in spring and fall"
    assert species_info.local_status([1] * 12)[0] == "Rarely reported here"


def test_expected_months_use_ten_percent_of_peak():
    counts = [100, 9, 10, 0, 0, 0, 0, 0, 0, 0, 50, 80]
    assert species_info.local_status(counts)[1] == [1, 3, 11, 12]


def test_links_for_birds_and_mammals():
    bird = species_info.links("Anna's Hummingbird", True, "annhum", 6317, "https://en.wikipedia.org/wiki/X")
    urls = {l["label"]: l["url"] for l in bird}
    assert urls["Audubon"].endswith("/annas-hummingbird")
    assert urls["All About Birds"].endswith("/Anna%27s_Hummingbird")
    assert urls["eBird"] == "https://ebird.org/species/annhum"
    mammal = species_info.links("Common Raccoon", False, None, 41663, None)
    assert [l["label"] for l in mammal] == ["iNaturalist"]


def test_get_builds_once_then_uses_cache(conn, fake_build):
    assert species_info.get(conn, "Bushtit")["sci_name"] == "Psaltriparus minimus"
    species_info.get(conn, "Bushtit")
    assert fake_build == ["Bushtit"]


def test_get_refreshes_stale_card(conn, fake_build):
    old = (datetime.now() - timedelta(days=45)).isoformat()
    db.save_species_info(conn, "Bushtit", {**CARD, "status": "old"}, old)
    assert species_info.get(conn, "Bushtit")["status"] == "Year-round resident"
    assert fake_build == ["Bushtit"]


def test_failed_refresh_keeps_old_card(conn, monkeypatch):
    old = (datetime.now() - timedelta(days=45)).isoformat()
    db.save_species_info(conn, "Bushtit", CARD, old)
    monkeypatch.setattr(species_info, "build", lambda name: (_ for _ in ()).throw(RuntimeError("offline")))
    assert species_info.get(conn, "Bushtit")["common"] == "Bushtit"


def test_blank_result_is_not_cached(conn, monkeypatch):
    monkeypatch.setattr(species_info, "build", lambda name: {"name": name, "links": []})
    assert species_info.get(conn, "Mystery Bird") is None
    assert db.get_species_info(conn, "Mystery Bird") is None


def test_no_card_for_none_or_other(conn, fake_build):
    assert species_info.get(conn, "None") is None and species_info.get(conn, "Other") is None
    assert fake_build == []


def test_month_counts(conn, add_visit):
    add_visit(species="Bushtit", captured_at="2026-01-05T09:00:00")
    add_visit(species="Bushtit", captured_at="2025-01-20T09:00:00")
    add_visit(species="Bushtit", captured_at="2026-10-01T09:00:00")
    add_visit(visitor="none", species=None, captured_at="2026-10-01T09:00:00")
    counts = db.month_counts(conn, "Bushtit")
    assert counts[0] == 2 and counts[9] == 1 and sum(counts) == 3


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


def test_card_route_renders_full_card(client, add_visit, fake_build):
    add_visit(species="Bushtit", captured_at="2026-01-05T09:00:00")
    html = client.get("/species-info/Bushtit").text
    assert "Psaltriparus minimus" in html and "Year-round resident" in html
    assert "AI summary based on Wikipedia" in html and "Jump to visits" in html and "1</span> visit" in html
    assert 'data-taxon="14886"' in html and "ebird.org/species/bushti" in html


def test_card_route_mini(client, fake_build):
    html = client.get("/species-info/Bushtit?mini=1").text
    assert "Click for more" in html and "<h2>" not in html


def test_card_fold_only_when_asked(client, add_visit, fake_build):
    add_visit(species="Bushtit")
    assert "<details" not in client.get("/species-info/Bushtit").text
    assert "<h2>" in client.get("/species-info/Bushtit").text
    folded = client.get("/species-info/Bushtit?fold=1").text
    assert '<details class="sp-more">' in folded and "More about this species" in folded
    assert folded.index("sp-more") > folded.index("sp-top") and folded.index('class="sp-map"') > folded.index("sp-more")
    assert "Click for more" in client.get("/species-info/Bushtit?mini=1&fold=1").text
    assert 'id="daily"' in client.get("/species-info/Bushtit?fold=1").text
    assert 'id="daily"' not in client.get("/species-info/Bushtit").text  # not in the pop-up


def test_species_page_folds_and_names_once(client, add_visit, fake_build):
    add_visit(species="Bushtit")
    client.get("/species-info/Bushtit")  # builds and caches the card
    html = client.get("/species/Bushtit").text
    assert '<details class="sp-more">' in html and html.count("<h1") == 1 and "<h2>" not in html
    assert "?fold=1" not in html  # rendered inline, no htmx load
    assert html.count('id="daily"') == 1 and html.index('id="daily"') > html.index("sp-more")
    assert html.count('id="hourly"') == 1 and html.index('id="hourly"') > html.index("sp-more")


def test_species_page_loading_card_has_no_title_and_asks_for_fold(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/species/Bushtit").text  # no cached card yet: loads over htmx
    assert "<h1>" not in html and "/species-info/Bushtit?fold=1" in html
    assert 'id="daily"' not in html  # the charts come with the card


def test_species_page_without_card_keeps_title(client, add_visit):
    add_visit(species="Other")
    html = client.get("/species/Other").text
    assert html.count("<h1>Other</h1>") == 1 and 'id="daily"' in html and "sp-more" not in html


def test_card_route_unknown(client, monkeypatch):
    monkeypatch.setattr(species_info, "build", lambda name: {"name": name, "links": []})
    assert "No info found" in client.get("/species-info/Nothing").text


def test_names_open_cards_but_not_false_triggers(client, add_visit):
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    assert 'data-species="Bushtit"' in client.get("/").text
    assert "data-species" not in client.get("/false-triggers").text


HOME = [45.78, -122.48]


def rec(id, type, q="A", lat=45.5, lon=-122.6, **extra):
    return {"id": str(id), "type": type, "q": q, "lat": str(lat), "lon": str(lon), "rec": f"Rec {id}",
            "lic": "//creativecommons.org/licenses/by-nc-sa/4.0/", "file": f"//xeno-canto.org/{id}/download",
            "loc": "Somewhere", "cnt": "United States", "length": "0:20", **extra}


def test_license_name():
    assert species_info.license_name("//creativecommons.org/licenses/by-nc-sa/4.0/") == "CC BY-NC-SA 4.0"
    assert species_info.license_name("") == ""


def test_pick_sounds_prefers_nearby_song_and_call():
    recs = [
        rec(1, "song", q="A", lat=40.7, lon=-74.0),   # New York, best quality but far
        rec(2, "song", q="B"),                        # near Portland
        rec(3, "call", q="C"),
        rec(4, "call, flight call", q="A", lat=47.6, lon=-122.3),  # Seattle, ~210 km
        rec(5, "alarm call", q="A", lat=None),        # no location
        {"id": "6", "type": "song"},                  # no file: skipped
    ]
    picked = species_info.pick_sounds(recs, HOME)
    assert [p["id"] for p in picked] == ["2", "3", "4"]  # the close call beats the better one 210 km away
    assert picked[0]["label"] == "Song" and picked[1]["label"] == "Call"
    assert picked[0]["file"] == "https://xeno-canto.org/2/download"
    assert picked[0]["page"] == "https://xeno-canto.org/2"
    assert picked[1]["km"] < 150 and 150 < picked[2]["km"] < 300 and picked[0]["license"] == "CC BY-NC-SA 4.0"


def test_pick_sounds_without_home_uses_quality():
    picked = species_info.pick_sounds([rec(1, "song", q="C"), rec(2, "song", q="A")], None, n=1)
    assert picked[0]["id"] == "2" and picked[0]["km"] is None


def test_cached_bird_card_gains_sounds_once_key_is_set(conn, monkeypatch):
    db.save_species_info(conn, "Bushtit", CARD, datetime.now().isoformat())
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "test-key")
    calls = []

    def fake_fetch(client, sci, center):
        calls.append(sci)
        return [{"id": "9", "label": "Song", "file": "https://x/9", "page": "https://xeno-canto.org/9",
                 "recordist": "A Recordist", "license": "CC BY 4.0", "where": "Oregon", "km": 12, "length": "0:30"}]
    monkeypatch.setattr(species_info, "fetch_sounds", fake_fetch)
    assert species_info.get(conn, "Bushtit")["sounds"][0]["id"] == "9"
    species_info.get(conn, "Bushtit")
    assert calls == ["Psaltriparus minimus"]  # stored, not fetched again


def test_no_sounds_without_key(conn, monkeypatch):
    db.save_species_info(conn, "Bushtit", CARD, datetime.now().isoformat())
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "")
    monkeypatch.setattr(species_info, "fetch_sounds", lambda *a: pytest.fail("should not call xeno-canto"))
    assert "sounds" not in species_info.get(conn, "Bushtit")


def test_card_shows_sounds_with_credits(client, monkeypatch):
    sounds = [{"id": "9", "label": "Song", "file": "https://xeno-canto.org/9/download", "page": "https://xeno-canto.org/9",
               "recordist": "A Recordist", "license": "CC BY 4.0", "where": "Portland, United States", "km": 1234,
               "length": "0:30"}]
    monkeypatch.setattr(species_info, "build", lambda name: {**CARD, "name": name, "sounds": sounds})
    html = client.get("/species-info/Bushtit").text
    assert '<audio controls preload="none" src="https://xeno-canto.org/9/download">' in html
    assert "A Recordist, CC BY 4.0" in html and "(1,234 km away)" in html and "XC9" in html
    assert "Recordings streamed from xeno-canto" in html


class FakeXC:
    """Stands in for httpx.Client: answers xeno-canto searches from a dict keyed by whether a box was used."""

    def __init__(self, boxed, everywhere):
        self.boxed, self.everywhere, self.queries = boxed, everywhere, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.queries.append(params["query"])
        recs = self.boxed if "box:" in params["query"] else self.everywhere
        return httpx.Response(200, json={"recordings": recs}, request=httpx.Request("GET", url))


def test_fetch_sounds_searches_near_home_first(monkeypatch):
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "k")
    fake = FakeXC([rec(1, "song"), rec(2, "call"), rec(3, "call")], [rec(9, "song")])
    picked = species_info.fetch_sounds(fake, "Junco hyemalis", HOME)
    assert [p["id"] for p in picked] == ["1", "2", "3"]
    assert len(fake.queries) == 1 and "box:41.28,-128.94,50.28,-116.02" in fake.queries[0]


def test_fetch_sounds_falls_back_when_few_nearby(monkeypatch):
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "k")
    fake = FakeXC([rec(1, "song")], [rec(1, "song"), rec(7, "call", lat=40.7, lon=-74.0)])
    picked = species_info.fetch_sounds(fake, "Junco hyemalis", HOME)
    assert [p["id"] for p in picked] == ["1", "7"] and len(fake.queries) == 2


def test_pick_sounds_closer_band_beats_quality_and_call_slot_is_a_call():
    recs = [
        rec(1, "song", q="A", lat=49.3, lon=-123.2),   # Vancouver BC, ~400 km
        rec(2, "song", q="B"),                         # Portland, ~30 km
        rec(3, "call, song", q="A"),
        rec(4, "call", q="C"),
    ]
    picked = species_info.pick_sounds(recs, HOME)
    assert [(p["id"], p["label"]) for p in picked] == [("2", "Song"), ("4", "Call"), ("3", "Song and call")]


SUMMER = {"local_counts": [0, 0, 0, 20, 40, 50, 50, 40, 20, 0, 0, 0], "expected_months": [4, 5, 6, 7, 8, 9]}
WINTER = {"local_counts": [50, 40, 20, 0, 0, 0, 0, 0, 0, 20, 40, 50], "expected_months": [1, 2, 3, 10, 11, 12]}


@pytest.mark.parametrize("info,month,badge", [
    (SUMMER, 6, None),
    (SUMMER, 3, "Early arrival"),
    (SUMMER, 10, "Late departure"),
    (SUMMER, 12, "Unusual this time of year"),
    (WINTER, 9, "Early arrival"),          # wraps: season runs Oct to Mar
    (WINTER, 4, "Late departure"),
    (WINTER, 7, "Unusual this time of year"),
    (WINTER, 1, None),
    ({"local_counts": [1] * 12, "expected_months": []}, 5, "Rarely reported here"),
    ({"name": "x"}, 5, None),              # no local data
    (None, 5, None),
])
def test_season_badge(info, month, badge):
    assert species_info.season_badge(info, month) == badge


def test_cards_show_season_badge_from_cached_card(client, add_visit, conn):
    db.save_species_info(conn, "Rufous Hummingbird", {**CARD, **SUMMER, "name": "Rufous Hummingbird"},
                         datetime.now().isoformat())
    add_visit(species="Rufous Hummingbird", captured_at="2026-03-28T09:00:00")
    add_visit(species="Bushtit", captured_at="2026-03-28T10:00:00")  # no cached card: no badge
    home = client.get("/").text
    assert home.count('class="badge out-of-season"') == 1 and "Early arrival" in home
    visit_id = db.list_visits(conn, species="Rufous Hummingbird")[0]["id"]
    page = client.get(f"/visit/{visit_id}").text
    assert "Early arrival" in page and "March is outside its usual months here." in page


def test_false_trigger_cards_get_no_season_badge(client, add_visit, conn):
    db.save_species_info(conn, "Rufous Hummingbird", {**CARD, **SUMMER}, datetime.now().isoformat())
    add_visit(visitor="none", species=None, corrected_species=None, captured_at="2026-03-28T09:00:00")
    assert 'badge out-of-season' not in client.get("/false-triggers").text


def test_short_status_for_tight_lists(monkeypatch):
    from birdbrain import species_info
    monkeypatch.setattr(species_info, "cached_cards", lambda: {"Bushtit": {"status": "Year-round resident"}, "Raccoon": {}})
    assert species_info.short_status("Bushtit") == ("Year-round", "Year-round resident")
    assert species_info.short_status("Raccoon") is None
    assert species_info.short_status("None") is None
