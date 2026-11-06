import re
from datetime import timedelta

import httpx

import pytest
from fastapi.testclient import TestClient

from birdbrain import db, species_info
from birdbrain.web import app as webapp
from helpers.cards import CARD, SOUNDS
from helpers.clock import FROZEN_NOW


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
    old = (FROZEN_NOW - timedelta(days=45)).isoformat()
    db.save_species_info(conn, "Bushtit", {**CARD, "status": "old"}, old)
    assert species_info.get(conn, "Bushtit")["status"] == "Year-round resident"
    assert fake_build == ["Bushtit"]


def test_failed_refresh_keeps_old_card(conn, monkeypatch):
    old = (FROZEN_NOW - timedelta(days=45)).isoformat()
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
    assert "AI summary based on Wikipedia" in html and "1</span> visit" in html
    assert 'data-taxon="14886"' in html and "ebird.org/species/bushti" in html


def test_card_route_mini(client, add_visit, fake_build):
    add_visit(species="Bushtit")
    client.get("/species-info/Bushtit")  # builds the card; the hover preview only reads it
    html = client.get("/species-info/Bushtit?mini=1").text
    assert "Click for more" in html and "<h2>" not in html


def test_card_fold_splits_into_the_bird_and_the_info_body(client, add_visit, fake_build):
    add_visit(species="Bushtit")
    assert "<details" not in client.get("/species-info/Bushtit").text
    assert "<h2>" in client.get("/species-info/Bushtit").text
    folded = client.get("/species-info/Bushtit?fold=1").text
    # The title and photo swap into the page; the Info tab's body rides along out of band.
    assert "sp-hero" in folded and folded.count("<h1") == 1 and 'id="sp-info-body" hx-swap-oob="innerHTML"' in folded
    assert folded.index('id="sp-info-body"') > folded.index("sp-photo")
    assert folded.index('class="sp-map"') > folded.index('id="sp-info-body"')
    assert "Click for more" in client.get("/species-info/Bushtit?mini=1&fold=1").text
    assert 'id="daily"' in folded and 'id="daily"' not in client.get("/species-info/Bushtit").text  # not in the pop-up
    assert "sp-more" not in folded and "Jump to visits" not in folded


def test_species_page_has_info_and_visits_tabs(client, add_visit, fake_build):
    add_visit(species="Bushtit")
    client.get("/species-info/Bushtit")  # builds and caches the card
    html = client.get("/species/Bushtit").text
    assert 'role="tablist"' in html and html.count('role="tab"') == 2 and html.count('role="tabpanel"') == 2
    assert html.index('data-sp-tab="visits"') < html.index('data-sp-tab="info"')
    assert re.search(r'id="tab-visits"[^>]*aria-selected="true"', html) and re.search(r'id="tab-info"[^>]*aria-selected="false" tabindex="-1"', html)  # Visits is the default tab
    assert re.search(r'id="panel-info"[^>]*hidden>', html) and not re.search(r'id="panel-visits"[^>]*hidden>', html) and "Jump to visits" not in html
    # Facts, charts and the card's sections live in the Info panel; the grid and its tools in the Visits panel.
    visits, info = html.index('id="panel-visits"'), html.index('id="panel-info"')
    assert "First seen" not in html
    assert "grid-count" not in html and html.index('class="tab-count"') < visits < html.index('class="lb-open"', visits) < info  # the count is in the Visits tab only
    assert re.search(r'data-sp-tab="info">Info</button>\s*<button[^>]*lb-tab', html)  # phones: count in the tab, expand button after it
    lb_tab = re.search(r'<button[^>]*lb-tab[^>]*>', html).group(0)
    assert "aria-disabled" not in lb_tab and "tabindex" not in lb_tab  # Visits is the default tab: the expand button starts enabled
    assert html.index("Year-round resident") < html.index('role="tablist"')  # the status pill sits with the bird
    assert info < html.index('id="daily"') and info < html.index('class="sp-map"')
    assert visits < html.index('class="grid-tools"') < html.index('id="visit-grid"') < info
    assert html.index("<h1") < html.index('role="tablist"')  # the bird comes first
    assert "species-tabs.js" in html and html.count("<h1") == 1 and "<h2>" not in html
    assert "?fold=1" not in html  # rendered inline, no htmx load
    assert html.count('id="daily"') == 1 and html.count('id="hourly"') == 1


def test_species_page_loading_card_asks_for_fold_and_keeps_a_fallback(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/species/Bushtit").text  # no cached card yet: loads over htmx
    assert "/species-info/Bushtit?fold=1" in html
    # Title, facts and charts stand in until the card arrives (the page drops them then), or stay if it never does.
    assert 'id="sp-fallback"' in html and 'id="daily"' in html


def test_species_page_without_card_keeps_title(client, add_visit):
    add_visit(species="Other")
    html = client.get("/species/Other").text
    assert html.count("<h1>Other</h1>") == 1 and 'id="daily"' in html and 'role="tablist"' in html
    assert html.index("<h1>Other</h1>") < html.index('role="tablist"') < html.index('id="visit-grid"') < html.index('id="daily"')


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
    db.save_species_info(conn, "Bushtit", CARD, FROZEN_NOW.isoformat())
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
    db.save_species_info(conn, "Bushtit", CARD, FROZEN_NOW.isoformat())
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "")
    monkeypatch.setattr(species_info, "fetch_sounds", lambda *a: pytest.fail("should not call xeno-canto"))
    assert "sounds" not in species_info.get(conn, "Bushtit")


def test_card_shows_sounds_with_credits(client, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    sounds = [{"id": "9", "label": "Song", "file": "https://xeno-canto.org/9/download", "page": "https://xeno-canto.org/9",
               "recordist": "A Recordist", "license": "CC BY 4.0", "where": "Portland, United States", "km": 1234,
               "length": "0:30"}]
    monkeypatch.setattr(species_info, "build", lambda name: {**CARD, "name": name, "sounds": sounds})
    html = client.get("/species-info/Bushtit").text
    assert '<audio controls preload="none" src="https://xeno-canto.org/9/download" data-card="Bushtit">' in html
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


def test_xeno_canto_errors_never_log_the_api_key(monkeypatch, caplog):
    """The key rides in the query string; a failed call must not put the URL in the journal."""
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "sekrit-key")

    class Denied:
        def get(self, url, params=None, headers=None, timeout=None):
            req = httpx.Request("GET", url, params=params)
            return httpx.Response(401, json={"error": "no"}, request=req)

    with pytest.raises(RuntimeError) as err:
        species_info.fetch_sounds(Denied(), "Junco hyemalis", HOME)
    assert "401" in str(err.value) and "sekrit-key" not in str(err.value) and "sekrit-key" not in repr(err.value.__cause__)
    monkeypatch.setattr(species_info, "httpx", species_info.httpx)
    monkeypatch.setattr(species_info.httpx, "Client", lambda **kw: _Ctx(Denied()))
    with caplog.at_level("DEBUG"):
        with db.session() as conn:
            db.init_db(conn)
            species_info._add_sounds(conn, "Dark-eyed Junco", {"is_bird": True, "sci_name": "Junco hyemalis"}, "2099-01-01T00:00:00")
    assert "xeno-canto lookup failed" in caplog.text and "401" in caplog.text and "sekrit-key" not in caplog.text


class _Ctx:
    def __init__(self, inner):
        self.inner = inner

    def __enter__(self):
        return self.inner

    def __exit__(self, *a):
        return False


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
                         FROZEN_NOW.isoformat())
    add_visit(species="Rufous Hummingbird", captured_at="2026-03-28T09:00:00")
    add_visit(species="Bushtit", captured_at="2026-03-28T10:00:00")  # no cached card: no badge
    home = client.get("/").text
    assert home.count('class="badge out-of-season"') == 1 and "Early arrival" in home
    visit_id = db.list_visits(conn, species="Rufous Hummingbird")[0]["id"]
    page = client.get(f"/visit/{visit_id}").text
    assert "Early arrival" in page and "March is outside its usual months here." in page


def test_false_trigger_cards_get_no_season_badge(client, add_visit, conn):
    db.save_species_info(conn, "Rufous Hummingbird", {**CARD, **SUMMER}, FROZEN_NOW.isoformat())
    add_visit(visitor="none", species=None, corrected_species=None, captured_at="2026-03-28T09:00:00")
    assert 'badge out-of-season' not in client.get("/false-triggers").text


def test_short_status_for_tight_lists(monkeypatch):
    from birdbrain import species_info
    monkeypatch.setattr(species_info, "cached_cards", lambda: {"Bushtit": {"status": "Year-round resident"}, "Raccoon": {}})
    assert species_info.short_status("Bushtit") == ("Year-round", "Year-round resident")
    assert species_info.short_status("Raccoon") is None
    assert species_info.short_status("None") is None


# --- a recording that won't play: the card is flagged, then its sounds are fetched again ---

def _save_card(conn, **extra):
    db.save_species_info(conn, "Bushtit", {**CARD, "sounds": [dict(s) for s in SOUNDS], **extra},
                         FROZEN_NOW.isoformat())


def _post(client, url, name="Bushtit", **kw):
    return client.post(f"/species-info/{name}/sound-broken", json={"url": url}, **kw)


def test_sound_broken_route_flags_only_a_real_sound_of_a_known_species(client, conn, add_visit):
    add_visit(species="Bushtit")
    _save_card(conn)
    assert _post(client, SOUNDS[0]["file"], name="Made Up Bird").status_code == 400
    assert _post(client, "https://elsewhere.example/x.mp3").status_code == 400
    assert client.post("/species-info/Bushtit/sound-broken", content="nope").status_code == 400
    assert not db.get_species_info(conn, "Bushtit")[0].get("sounds_broken")
    assert _post(client, SOUNDS[0]["file"]).status_code == 204
    card = db.get_species_info(conn, "Bushtit")[0]
    assert card["sounds_broken"] is True and len(card["sounds"]) == 2  # flagged, nothing removed


def test_sound_broken_allowed_for_visitors_on_the_pi_but_not_the_mirror(client, conn, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    _save_card(conn)
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    assert _post(client, SOUNDS[1]["file"]).status_code == 204
    assert client.post("/species-info/Bushtit/delete").status_code == 403  # nothing else opened up
    assert _post(client, SOUNDS[1]["file"], headers={"sec-fetch-site": "cross-site"}).status_code == 403
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    assert _post(client, SOUNDS[1]["file"]).status_code == 404  # the mirror takes no POST but the guest sign-in


def test_flagged_card_refetches_sounds_and_clears_the_flag(conn, monkeypatch):
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "k")
    _save_card(conn, sounds_broken=True)
    new = [{"label": "Song", "file": "https://xeno-canto.org/3/download", "id": 3}]
    monkeypatch.setattr(species_info, "fetch_sounds", lambda *a: new)
    assert species_info.get(conn, "Bushtit", fetch=False)["sounds"][0]["id"] == 1  # viewers never trigger it
    card = species_info.get(conn, "Bushtit")
    assert card["sounds"] == new and "sounds_broken" not in db.get_species_info(conn, "Bushtit")[0]


def test_empty_refetch_still_clears_the_flag(conn, monkeypatch):
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "k")
    _save_card(conn, sounds_broken=True)
    monkeypatch.setattr(species_info, "fetch_sounds", lambda *a: [])
    assert species_info.get(conn, "Bushtit")["sounds"] == []
    assert "sounds_broken" not in db.get_species_info(conn, "Bushtit")[0]


def test_failed_refetch_keeps_sounds_and_flag_and_backs_off(conn, monkeypatch):
    monkeypatch.setattr(species_info.config, "XC_API_KEY", "k")
    _save_card(conn, sounds_broken=True)
    calls = []

    def boom(*a):
        calls.append(1)
        raise httpx.ConnectError("down")
    monkeypatch.setattr(species_info, "fetch_sounds", boom)
    for _ in range(3):
        card = species_info.get(conn, "Bushtit")
    assert len(calls) == 1 and len(card["sounds"]) == 2 and card["sounds_broken"] is True
    species_info._failed_at["sounds:Bushtit"] -= species_info.RETRY_AFTER + 1
    species_info.get(conn, "Bushtit")
    assert len(calls) == 2

# --- header facts (one Haiku call with the notes) ---

class FakeHaiku:
    """identify.client() stand-in: parse() returns the scripted CardText and keeps the request."""

    def __init__(self, notes, facts):
        self.beta = self
        self.messages = self
        self.parsed = species_info.CardText(notes=notes, facts=facts)
        self.requests = []

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        return type("R", (), {"parsed_output": self.parsed})()


def F(lead, text):
    return {"lead": lead, "text": text}


def test_clean_facts_guard():
    notes = ["Eats seeds on the ground.", "Travels in flocks in winter."]
    facts = [
        F("x", ""),                                                           # blank
        F("x", "   "),                                                        # blank
        F("Quarters", "Weighs about as much as three quarters."),             # kept
        F("", "Eats seeds on the ground."),                                   # repeats a note
        F("Flocks", "Flocks travel together in winter."),                     # mostly the same words as a note
        F("", " ".join(["word"] * 21)),                                       # too long
        F("Quarters", "Weighs about as much as three quarters!"),             # repeats an earlier fact
        F("Males", "Males sing from the tops of tall trees."),
        F("10 years", "Some live over 10 years in the wild."),
        F("Snow bird", "Its name means snow bird."),
        F("Pairs", "Pairs stay together all season."),
        F("Sixth", "Sixth good fact is cut off by the cap."),
    ]
    got = species_info.clean_facts(facts, notes)
    assert got[0] == F("Quarters", "Weighs about as much as three quarters.")
    texts = [f["text"] for f in got]
    assert "Eats seeds on the ground." not in texts and "Flocks travel together in winter." not in texts
    assert len(got) == species_info.MAX_FACTS
    assert not any(len(t.split()) > species_info.MAX_FACT_WORDS or not t.strip() for t in texts)
    assert species_info.clean_facts(None, notes) == []


def test_clean_facts_accepts_models_dicts_and_old_strings():
    out = species_info.clean_facts([species_info.Fact(lead="Blizzard", text="A flock is called a blizzard."),
                                    "Plain old string fact lives to 11 years."], [])
    assert out == [F("Blizzard", "A flock is called a blizzard."), F("11 years", "Plain old string fact lives to 11 years.")]


@pytest.mark.parametrize("lead, text, want", [
    ("Blizzard", "A flock is called a blizzard.", "Blizzard"),                      # valid, case-insensitive
    ("75%", "Seeds make up 75% of the diet.", "75%"),                               # number as written
    ("75%", "Seeds make up 75 percent of the diet.", "75%"),                        # % vs the word
    ("12 years", "Lives up to 12-year stretches.", "12 years"),                     # unit dropped in the text
    ("40 million", "About 40 million live in North America.", "40 million"),
    ("1,741", "Named in 1741 by a Swede.", "1,741"),                                # same number, comma or not
    ("1,800", "Named in 1741 by a Swede.", "1741"),                                 # a different number: the text's own
    ("", "Weighs about 20 grams.", "20 grams"),                                     # missing: number fallback with unit
    ("", "Lives in flocks near feeders.", ""),                                      # missing, no number: no lead
    ("Hummingbird", "A flock is called a blizzard.", ""),                           # not in the text, no number
    ("Warbler", "Weighs 9 ounces and sings.", "9 ounces"),                          # not in text: number fallback
    ("Seven word long lead here", "Seven word long lead here is odd.", ""),         # over 3 words
    ("Extraordinarily rare", "Extraordinarily rare bird.", ""),                          # over the character cap
    ("Eight", "About 8% of the diet is bugs.", "8%"),                               # word lead absent: number fallback
])
def test_clean_lead(lead, text, want):
    assert species_info.clean_lead(lead, text) == want


def test_clean_notes_drops_a_note_that_repeats_an_earlier_one():
    notes = ["In winter it often forages in flocks, and it is common at feeders.",
             "Juncos forage in mixed flocks in winter and are common at feeders.",
             "", "Eats seeds and small insects.", "Eats seeds and small insects."]
    assert species_info.clean_notes(notes) == [notes[0], "Eats seeds and small insects."]
    many = ["Eats beetles.", "Nests in hedges.", "Sings at dawn.", "Drinks often.", "Migrates south."]
    assert len(species_info.clean_notes(many)) == species_info.MAX_NOTES


def test_card_text_parses_both_lists_and_guards_them(monkeypatch):
    from birdbrain import identify
    Fact = species_info.Fact
    fake = FakeHaiku(["Eats seeds.", "", "Lives in flocks."] + [f"Note about {w}." for w in ("red", "blue", "green", "pink")],
                     [Fact(lead="Deck of cards", text="A tiny bird that weighs less than a deck of cards."),
                      Fact(lead="x", text="Eats seeds."), Fact(lead="", text="")])
    monkeypatch.setattr(identify, "client", lambda: fake)
    out = species_info.card_text("Bushtit", "Article text " * 10)
    assert len(out["notes"]) == species_info.MAX_NOTES and out["notes"][:2] == ["Eats seeds.", "Lives in flocks."]
    assert out["facts"] == [F("Deck of cards", "A tiny bird that weighs less than a deck of cards.")]
    sent = fake.requests[0]
    assert sent["output_format"] is species_info.CardText and "facts" in sent["system"] and "lead" in sent["system"]
    assert "Animal: Bushtit" in sent["messages"][0]["content"]


def test_card_text_without_a_parsed_answer(monkeypatch):
    from birdbrain import identify
    fake = FakeHaiku([], [])
    fake.parsed = None
    monkeypatch.setattr(identify, "client", lambda: fake)
    assert species_info.card_text("Bushtit", "x") == {"notes": [], "facts": []}


class FakeWiki:
    """httpx.Client stand-in for the two Wikipedia calls (summary, then plain text)."""

    def __init__(self, fail=(), limited=None):
        # limited: {url fragment: [(status, headers), ...]}, one scripted reply per call before the real one
        self.urls, self.headers_seen, self.fail, self.limited = [], [], fail, limited or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None, headers=None, timeout=None):
        self.urls.append(url)
        self.headers_seen.append(headers)
        req = httpx.Request("GET", url)
        for frag, script in self.limited.items():
            if frag in url and script:
                status, headers = script.pop(0)
                return httpx.Response(status, request=req, headers=headers)
        if any(f in url for f in self.fail):
            return httpx.Response(500, request=req)
        if "rest_v1" in url:
            return httpx.Response(200, request=req, json={"type": "standard", "title": "Bushtit"})
        return httpx.Response(200, request=req, json={"query": {"pages": {"1": {"extract": "The bushtit is tiny."}}}})


def cached_card(conn, name, **extra):
    card = {**CARD, "name": name, "common": name, "wiki_url": "https://en.wikipedia.org/wiki/" + name.replace(" ", "_"), **extra}
    db.save_species_info(conn, name, card, "2026-10-01T00:00:00")
    return card


def test_backfill_facts_updates_only_notes_and_facts(conn, monkeypatch):
    from birdbrain import identify
    cached_card(conn, "Bushtit")
    cached_card(conn, "Has Facts", facts=["Already here."])
    cached_card(conn, "No Wiki", wiki_url=None)
    fake = FakeHaiku(["New note."], [species_info.Fact(lead="Deck of cards", text="Weighs less than a deck of cards.")])
    monkeypatch.setattr(identify, "client", lambda: fake)
    wiki = FakeWiki()
    monkeypatch.setattr(species_info.httpx, "Client", lambda **kw: wiki)
    lines = []
    assert species_info.backfill_facts(conn, out=lines.append) == (1, 0)
    card, stamp = db.get_species_info(conn, "Bushtit")
    assert card["facts"] == [F("Deck of cards", "Weighs less than a deck of cards.")] and card["notes"] == ["New note."]
    assert card["photo"] == CARD["photo"] and card["description"] == CARD["description"]  # nothing else touched
    assert stamp == "2026-10-01T00:00:00"
    assert db.get_species_info(conn, "Has Facts")[0]["facts"] == ["Already here."]
    assert "facts" not in db.get_species_info(conn, "No Wiki")[0]
    assert lines == ["1/1 done, 0 failed"] and len(fake.requests) == 1
    assert all(h["Accept-Encoding"] == "gzip" and "Birdbrain" in h["User-Agent"] for h in wiki.headers_seen)
    # --redo takes the card that already has facts too
    assert species_info.backfill_facts(conn, redo=True, only="has facts", out=lines.append) == (1, 0)


def test_backfill_facts_dry_run_and_failure_keep_old_cards(conn, monkeypatch):
    from birdbrain import identify
    cached_card(conn, "Bushtit")
    monkeypatch.setattr(identify, "client", lambda: FakeHaiku(["n"], [species_info.Fact(lead="", text="Weighs less than a deck of cards.")]))
    lines = []
    assert species_info.backfill_facts(conn, dry_run=True, out=lines.append) == (0, 0)
    assert "Bushtit" in lines[0] and "facts" not in db.get_species_info(conn, "Bushtit")[0]
    monkeypatch.setattr(species_info.httpx, "Client", lambda **kw: FakeWiki(fail=("w/api.php",)))
    lines.clear()
    assert species_info.backfill_facts(conn, out=lines.append) == (0, 1)
    assert lines == ["1/1 done, 1 failed"]
    assert db.get_species_info(conn, "Bushtit")[0]["notes"] == CARD["notes"]  # the old card survives


def test_rebuild_keeps_old_facts_when_new_build_has_none(conn, monkeypatch):
    cached = ({**CARD, "facts": ["Old fact that stays."]}, "2026-01-01T00:00:00")
    monkeypatch.setattr(species_info, "build", lambda name: {"name": name, "inat_id": 1, "links": []})
    out = species_info._rebuild(conn, "Bushtit", cached)
    assert out["facts"] == ["Old fact that stays."]


def test_header_play_buttons_carry_the_card_name(client, conn, add_visit):
    """The header's Song/Call buttons say which species they belong to, so a clip that won't play can be reported."""
    add_visit(species="Bushtit")
    _save_card(conn)
    card = db.get_species_info(conn, "Bushtit")
    for sound in card[0]["sounds"]:
        sound.update(km=100, recordist="A")
    db.save_species_info(conn, "Bushtit", card[0], card[1])
    html = client.get("/species-info/Bushtit?fold=1").text
    buttons = re.findall(r'<button[^>]*class="sp-play"[^>]*>', html)
    assert len(buttons) == 2 and all('data-card="Bushtit"' in b for b in buttons)


HOOKS = [F("Blizzard", "A flock of juncos is called a blizzard."), F("75%", "Seeds make up 75% of the diet."),
         F("Arctic", "Some breed as far north as the Arctic."), F("1741", "Named by Linnaeus in 1741."),
         F("", "Old tile with no lead lives in flocks.")]


def _hero(client, conn, add_visit, facts):
    add_visit(species="Bushtit")
    _save_card(conn, facts=facts, sounds=[{**s, "km": 100, "recordist": "A"} for s in SOUNDS])
    return client.get("/species-info/Bushtit?fold=1").text


def test_species_header_shows_a_tile_per_fact_with_its_lead(client, conn, add_visit):
    html = _hero(client, conn, add_visit, HOOKS[:4])
    assert len(re.findall(r'class="fact-tile[ "]', html)) == 4 and 'data-n="4"' in html
    for lead in ("Blizzard", "75%", "Arctic", "1741"):
        assert f"<b>{lead}</b>" in html
    assert 'data-tip="AI summary based on Wikipedia"' in html
    assert "has-facts" in html
    # the sound buttons move under the name, inside the name block (before the tiles)
    assert html.index("sp-hero-text") < html.index("sp-plays") < html.index("fact-tiles")


def test_species_header_five_facts_and_a_leadless_tile(client, conn, add_visit):
    html = _hero(client, conn, add_visit, HOOKS)
    assert len(re.findall(r'class="fact-tile[ "]', html)) == 5 and html.count("<b>") == 4 and "no-lead" in html


def test_species_header_old_string_facts_still_render(client, conn, add_visit):
    html = _hero(client, conn, add_visit, ["Old plain fact one here.", "Old plain fact two here."])
    assert len(re.findall(r'class="fact-tile[ "]', html)) == 2 and "<b>" not in html and "Old plain fact two here." in html


def test_species_header_without_facts_is_as_before(client, conn, add_visit):
    for facts in ([], HOOKS[:1]):  # one fact alone is not a tile group
        html = _hero(client, conn, add_visit, facts)
        assert "fact-tile" not in html.replace(".fact-tile", "") and "has-facts" not in html
        assert html.index("sp-hero-text") < html.index("sp-hero-id") + 400 and "sp-plays" in html


def test_pop_up_card_has_no_tiles(client, conn, add_visit):
    add_visit(species="Bushtit")
    _save_card(conn, facts=HOOKS[:4], sounds=[{**s, "km": 100, "recordist": "A"} for s in SOUNDS])
    assert "fact-tile" not in client.get("/species-info/Bushtit").text


@pytest.fixture
def naps(monkeypatch):
    """Replaces the backfill's sleep, so nothing waits; the list holds the seconds it was asked to wait."""
    waits = []
    monkeypatch.setattr(species_info, "_sleep", waits.append)
    return waits


def _scripted(monkeypatch, wiki):
    from birdbrain import identify
    monkeypatch.setattr(identify, "client", lambda: FakeHaiku(["n"], [species_info.Fact(lead="Deck", text="Weighs less than a deck of cards.")]))
    monkeypatch.setattr(species_info.httpx, "Client", lambda **kw: wiki)


def test_backfill_waits_out_a_429_then_succeeds(conn, monkeypatch, naps):
    cached_card(conn, "Bushtit")
    wiki = FakeWiki(limited={"rest_v1": [(429, {}), (503, {})]})  # then the real reply; no Retry-After: 10 s then 30 s
    _scripted(monkeypatch, wiki)
    assert species_info.backfill_facts(conn, out=lambda m: None) == (1, 0)
    assert naps == [10, 30]
    assert sum("rest_v1" in u for u in wiki.urls) == 3


def test_backfill_honours_retry_after_and_gives_up_after_two_retries(conn, monkeypatch, naps):
    cached_card(conn, "Bushtit")
    wiki = FakeWiki(limited={"rest_v1": [(429, {"Retry-After": "7"}), (429, {"Retry-After": "9999"}), (429, {})]})
    _scripted(monkeypatch, wiki)
    lines = []
    assert species_info.backfill_facts(conn, out=lines.append) == (0, 1)
    assert naps == [7, species_info.RETRY_AFTER_MAX]  # capped; the third 429 is a failure, with no third wait
    assert lines == ["1/1 done, 1 failed"]
    assert "facts" not in db.get_species_info(conn, "Bushtit")[0]  # the old card survives


def test_backfill_pauses_between_cards(conn, monkeypatch, naps):
    for n in ("Bushtit", "Chickadee", "Wren"):
        cached_card(conn, n)
    _scripted(monkeypatch, FakeWiki())
    assert species_info.backfill_facts(conn, out=lambda m: None) == (3, 0)
    assert naps == [species_info.PACE, species_info.PACE]  # between cards, not before the first or after the last


def test_clean_facts_drops_a_leadless_fact_unless_too_few_remain():
    notes = []
    mixed = [F("Blizzard", "A flock of juncos is called a blizzard."), F("", "Juncos forage on the ground in winter flocks."),
             F("Arctic", "Some juncos breed as far north as the Arctic.")]
    assert [f["lead"] for f in species_info.clean_facts(mixed, notes)] == ["Blizzard", "Arctic"]
    one_led = [F("Blizzard", "A flock of juncos is called a blizzard."), F("", "Juncos forage on the ground in winter flocks.")]
    assert len(species_info.clean_facts(one_led, notes)) == 2  # dropping would leave 1: keep both
    none_led = [F("", "Juncos forage on the ground in winter flocks."), F("", "Males sing from the tops of tall trees.")]
    assert len(species_info.clean_facts(none_led, notes)) == 2
    many = [F("Beetles", "It outruns beetles on open ground."), F("Granite", "Nests are lined with crushed granite."),
            F("Violins", "Its song resembles violins tuning up."), F("Marathons", "Some migrate marathons every autumn."),
            F("Volcanoes", "Young birds roost near volcanoes."), F("Teapots", "Collectors prize teapots with its portrait."),
            F("Glaciers", "Glaciers shaped its breeding range.")]
    assert len(species_info.clean_facts(many, notes)) == species_info.MAX_FACTS


def test_prompt_asks_for_whole_single_claim_sentences_with_a_lead():
    p = species_info.NOTES_PROMPT
    assert "complete sentence" in p and "exactly one claim" in p and "Every fact must have a lead" in p


def _last_seen_visit(add_visit):
    from datetime import timedelta
    from helpers.clock import FROZEN_NOW
    add_visit(species="Bushtit", captured_at=(FROZEN_NOW - timedelta(days=2)).isoformat())


def test_hero_has_last_seen_line_under_the_songs(client, conn, add_visit):
    _last_seen_visit(add_visit)
    for facts in (HOOKS[:4], []):  # with tiles (inside the name block) and without
        _save_card(conn, facts=facts, sounds=[{**s, "km": 100, "recordist": "A"} for s in SOUNDS])
        html = client.get("/species-info/Bushtit?fold=1").text
        assert html.count('class="sp-last-seen') == 1
        assert html.index("sp-plays") < html.index("sp-last-seen")
        assert re.search(r'Last seen <span class="ago" data-bare="1" data-age="\d+">2 days ago</span>', html)


def test_hero_last_seen_line_without_sounds_sits_under_the_badge(client, conn, add_visit):
    _last_seen_visit(add_visit)
    _save_card(conn, facts=[], sounds=[])
    html = client.get("/species-info/Bushtit?fold=1").text
    assert "sp-plays" not in html and html.index("sp-status") < html.index("sp-last-seen") < html.index("sp-hero-row") + 4000


def test_info_tab_drops_last_seen_but_pop_up_keeps_it(client, conn, add_visit):
    _last_seen_visit(add_visit)
    _save_card(conn, facts=[], sounds=[])
    folded = client.get("/species-info/Bushtit?fold=1").text
    body = folded[folded.index('id="sp-info-body"'):]
    assert "Last seen" not in body and 'class="sp-facts' in body  # the confirm badge row stays
    popup = client.get("/species-info/Bushtit").text
    assert "sp-last-seen" not in popup
    assert re.search(r'Last seen <strong>[^<]+</strong> <span class="ago" data-age="\d+">\(2 days ago\)</span>', popup)
