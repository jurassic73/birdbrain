"""Component tests of birdbrain/species.py (info): one real local dependency, a temp database or files on disk."""

from datetime import timedelta

import httpx
import pytest

from helpers.cards import CARD
from helpers.clock import FROZEN_NOW
from helpers.species_info import F, HOME, SUMMER, FakeHaiku, fake_build, _save_card
from helpers.web import _card
from birdbrain import db, species_info


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
    full = [f"Fact number {i} is already here." for i in range(species_info.FACTS_WANTED)]
    cached_card(conn, "Has Facts", facts=full)
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
    assert db.get_species_info(conn, "Has Facts")[0]["facts"] == full  # a full pool is left alone
    assert "facts" not in db.get_species_info(conn, "No Wiki")[0]
    assert lines == ["1/1 done, 0 failed"] and len(fake.requests) == 1
    assert all(h["Accept-Encoding"] == "gzip" and "Birdbrain" in h["User-Agent"] for h in wiki.headers_seen)
    # --redo takes the card that already has facts too
    assert species_info.backfill_facts(conn, redo=True, only="has facts", out=lines.append) == (1, 0)


def test_backfill_facts_fills_a_thin_pool_without_redo(conn, monkeypatch):
    """Cards from before the pool have 2 to 5 facts: a plain --facts run brings them up, no --redo needed."""
    from birdbrain import identify
    cached_card(conn, "Thin", facts=[f"Old fact {i} stays put." for i in range(species_info.FACTS_WANTED - 1)])
    fake = FakeHaiku(["n"], [species_info.Fact(lead="Trill", text="Its song is a trill.", topic="song")])
    monkeypatch.setattr(identify, "client", lambda: fake)
    monkeypatch.setattr(species_info.httpx, "Client", lambda **kw: FakeWiki())
    lines = []
    assert species_info.backfill_facts(conn, dry_run=True, out=lines.append) == (0, 0) and "Thin" in lines[0]
    assert species_info.backfill_facts(conn, out=lines.append) == (1, 0)
    assert db.get_species_info(conn, "Thin")[0]["facts"] == [{"lead": "Trill", "text": "Its song is a trill.", "topic": "song"}]


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


def test_rebuild_never_shrinks_the_fact_pool(conn, monkeypatch):
    """The monthly rebuild asks Claude again; an answer with fewer facts must not replace a bigger pool."""
    pool = [f"Pool fact number {i}." for i in range(12)]
    cached = ({**CARD, "facts": pool, "notes": ["Old note."]}, "2026-01-01T00:00:00")
    fresh = {"name": "Bushtit", "inat_id": 1, "links": [], "description": "New text."}
    monkeypatch.setattr(species_info, "build", lambda name: {**fresh, "facts": ["Only one."], "notes": ["New note."]})
    out = species_info._rebuild(conn, "Bushtit", cached)
    assert out["facts"] == pool and out["notes"] == ["Old note."]  # the notes the pool was checked against stay too
    assert out["description"] == "New text."  # the rest of the card is still refreshed
    bigger = pool + ["A thirteenth."]
    monkeypatch.setattr(species_info, "build", lambda name: {**fresh, "facts": bigger, "notes": ["New note."]})
    out = species_info._rebuild(conn, "Bushtit", cached)
    assert out["facts"] == bigger and out["notes"] == ["New note."]


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


def test_cached_cards_are_read_on_a_freshly_booted_machine(conn, monkeypatch):
    """time.monotonic() counts from boot on Linux: with 0.0 as the "never read" mark the first minute after a reboot
    served no cards at all, so pages lost their season badges (and CI failed on a seconds-old runner VM)."""
    db.save_species_info(conn, "Rufous Hummingbird", {**CARD, **SUMMER}, FROZEN_NOW.isoformat())
    monkeypatch.setattr(species_info.time, "monotonic", lambda: 5.0)
    species_info.forget_cached_cards()
    assert species_info.cached_cards()["Rufous Hummingbird"]["expected_months"] == SUMMER["expected_months"]
    assert species_info.visit_badge("Rufous Hummingbird", "2026-03-28T09:00:00") == "Early arrival"


def test_cached_facts_returns_texts_for_both_formats(conn):
    from birdbrain import species_info
    _card(conn, "Bushtit", facts=[{"lead": "Flocks", "text": "Bushtits travel in noisy flocks."}, "Old string.", {"text": " "}, 5])
    assert species_info.cached_facts(conn, "Bushtit") == ["Bushtits travel in noisy flocks.", "Old string."]
    assert species_info.cached_facts(conn, "Bushtit", limit=1) == ["Bushtits travel in noisy flocks."]
    assert species_info.cached_facts(conn, "Other") == [] and species_info.cached_facts(conn, None) == []


def test_cached_facts_rotate_through_the_pool(conn):
    from birdbrain import species_info
    pool = [f"Pool fact number {i}." for i in range(7)]
    _card(conn, "Bushtit", facts=pool)
    assert species_info.cached_facts(conn, "Bushtit") == pool[:3]
    assert species_info.cached_facts(conn, "Bushtit", turn=1) == pool[3:6]
    assert species_info.cached_facts(conn, "Bushtit", turn=2) == [pool[6], pool[0], pool[1]]
