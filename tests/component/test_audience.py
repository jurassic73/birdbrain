"""birdbrain/audience with its SQLite file: flushing, the feed's rows, the Pi's idempotent merge and its pull (with
the HTTP call faked at the boundary), and the Guests page's sums."""
import time
from datetime import timedelta

import pytest

from helpers.clock import FROZEN_NOW
from birdbrain import audience

BROWSER = "Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Safari/605.1.15"


@pytest.fixture
def adb(tmp_path, monkeypatch):
    path = tmp_path / "state" / "audience.db"
    monkeypatch.setenv("AUDIENCE_DB", str(path))
    return path


def test_flush_adds_to_the_file_and_keeps_todays_names(adb):
    c = audience.counter
    now = time.time()
    c.view("home", "203.0.113.9", BROWSER, "NZ", False, now)
    audience.flush(now=now)
    c.view("home", "203.0.113.9", BROWSER, "NZ", False, now)
    audience.flush(now=now)
    rows = {m: n for _, m, n in audience.rows_since(0)}
    assert rows["page:home"] == 2 and rows["visitors"] == 1
    # A restart: a fresh counter picks up today's salt and names, so the same person isn't a new visitor.
    fresh = audience.Counter()
    audience.load(fresh)
    fresh.view("home", "203.0.113.9", BROWSER, "NZ", False, now)
    assert not any(m == "visitors" for _, m, _ in fresh.take())
    assert "203.0.113.9" not in adb.read_bytes().decode("latin-1")  # no address in the file


def test_the_mirror_forgets_rows_past_a_year_and_a_bit(adb):
    now = time.time()
    audience.merge([(int(now) - audience.MIRROR_KEEP - 600, "visitors", 3), (int(now) - 600, "visitors", 1)])
    audience.flush(now=now)
    assert [n for _, _, n in audience.rows_since(0)] == [1]


def test_merge_replaces_so_pulling_twice_changes_nothing(adb):
    a = int(time.time()) // 300 * 300 - 3600
    b = a + 300
    rows = [[a, "visitors", 3], [a, "page:home", 5], [b, "visitors", 1]]
    audience.merge(rows)
    audience.merge(rows)
    audience.merge([[b, "visitors", 2], ["bad"], [b, "x" * 100, 1], [b, "returning", -1]])  # a slot grew; junk
    assert sorted(map(tuple, audience.rows_since(0))) == [(a, "page:home", 5), (a, "visitors", 3), (b, "visitors", 2)]


class Reply:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("nope", request=None, response=None)

    def json(self):
        return self.payload


def test_pull_fetches_with_the_token_and_at_most_every_couple_of_minutes(adb, monkeypatch):
    calls = []
    slot = int(time.time()) // 300 * 300 - 3 * 86400

    def get(url, params, timeout, headers):
        calls.append((url, params, headers))
        return Reply({"rows": [[slot, "visitors", 4]]})
    import httpx
    monkeypatch.setattr(httpx, "get", get)
    audience.pull(now=5000.0)
    assert calls == []  # not configured: nothing asked
    monkeypatch.setenv("AUDIENCE_SOURCE_URL", "https://mirror.example.com/api/audience-counts")
    monkeypatch.setenv("AUDIENCE_TOKEN", "tok")
    audience.pull(now=5000.0)
    audience.pull(now=5030.0)  # too soon
    assert len(calls) == 1 and calls[0][2][audience.TOKEN_HEADER] == "tok" and calls[0][1] == {"since": 0}
    audience.pull(now=5000.0 + audience.PULL_EVERY)
    assert calls[1][1] == {"since": slot - audience.PULL_OVERLAP}  # the newest slot it has, less a day's overlap
    assert audience.rows_since(0) == [[slot, "visitors", 4]]


def test_a_failed_pull_is_noted_and_the_old_counts_stay(adb, monkeypatch):
    import httpx
    monkeypatch.setenv("AUDIENCE_SOURCE_URL", "https://mirror.example.com/api/audience-counts")
    monkeypatch.setenv("AUDIENCE_TOKEN", "tok")
    audience.merge([[1000, "visitors", 4]])
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Reply({}, 404))
    audience.pull(now=9000.0)
    r = audience.report("7d", FROZEN_NOW)
    assert r["pull_error"] and r["pulled_at"] is None and r["any_counts"]


def test_report_sums_the_period_and_compares_with_the_one_before(adb):
    now = FROZEN_NOW
    at = lambda dt: int(dt.timestamp()) // 300 * 300  # noqa: E731
    audience.merge([
        (at(now - timedelta(days=20)), "visitors", 1),  # counting began here, before the previous 7 days
        (at(now - timedelta(hours=1)), "visitors", 6), (at(now - timedelta(hours=1)), "page:home", 9),
        (at(now - timedelta(hours=1)), "page_people:home", 5), (at(now - timedelta(hours=1)), "country:NZ", 4),
        (at(now - timedelta(hours=1)), "country:XX", 2), (at(now - timedelta(hours=1)), "click:theme", 3),
        (at(now - timedelta(hours=1)), "click:forged", 50),
        (at(now - timedelta(days=8)), "visitors", 3), (at(now - timedelta(days=8)), "page:home", 2),
    ])
    r = audience.report("7d", now)
    head = {h["key"]: h for h in r["head"]}
    assert head["visitors"]["n"] == 6 and head["visitors"]["prev"] == 3 and head["visitors"]["change"] == 100
    assert head["views"]["n"] == 9
    assert [(p["name"], p["n"], p["people"]) for p in r["pages"]] == [("home", 9, 5)]
    assert [c["name"] for c in r["controls"]] == ["theme"]  # a name off the list is never shown
    assert [t["code"] for t in r["countries"]] == ["NZ", "XX"] and r["unknown"] == 1
    assert sum(v or 0 for v in r["chart"]["data"]) == 6 and len(r["chart"]["data"]) == 7
    assert r["map"] and any(p["code"] == "NZ" and p["step"] for p in r["map"]["paths"])
    today = audience.report("today", now)
    assert today["chart"]["data"][now.hour + 1] is None  # hours still to come are gaps
