"""Component tests of birdbrain/weather.py: one real local dependency, a temp database or files on disk."""

from datetime import date, datetime, timedelta

import httpx
import pytest

from helpers.weather import canned, located  # noqa: F401  (autouse fixture)
from birdbrain import config, db, weather


NOW = datetime(2026, 11, 2, 14, 30)


@pytest.fixture
def api(monkeypatch):
    """Stub the network: records each call; answers with canned data for the days it was asked for."""
    calls = []

    def fake(url, **params):
        calls.append((url, params))
        if url == weather.ARCHIVE_URL:
            return canned(params["start_date"], params["end_date"])
        first = (date(2026, 11, 2) - timedelta(days=params["past_days"])).isoformat()
        return canned(first, "2026-11-02", rain_on="2026-11-01")

    monkeypatch.setattr(weather, "_get", fake)
    return calls


def test_upsert_twice_has_no_duplicates(conn):
    days, hours = weather.parse(canned("2026-11-01", "2026-11-02"), "a")
    db.save_weather(conn, days, hours)
    days[0]["precip_mm"] = 3.0
    db.save_weather(conn, days, hours)
    assert conn.execute("SELECT COUNT(*) FROM weather_hours").fetchone()[0] == 48
    rows = db.weather_days(conn, "2026-11-01", "2026-11-02")
    assert [r["day"] for r in rows] == ["2026-11-01", "2026-11-02"] and rows[0]["precip_mm"] == 3.0
    assert len(db.weather_hours(conn, "2026-11-01", "2026-11-01")) == 24
    assert len(db.weather_hours(conn, "2026-11-01", "2026-11-02")) == 48


def test_refresh_backfills_then_only_refetches_open_days(conn, add_visit, api):
    add_visit(captured_at="2026-10-31T09:00:00")
    done = weather.refresh(conn, NOW)
    assert done == {"days": 3, "hours": 24 * 2 + 15, "first": "2026-10-31", "last": "2026-11-02"}
    assert len(api) == 1 and api[0][0] == weather.FORECAST_URL
    assert api[0][1]["past_days"] == 2 and api[0][1]["timezone"] == "auto"
    assert api[0][1]["latitude"] == 1.5 and "temperature_2m" in api[0][1]["hourly"]
    rows = {r["day"]: r for r in db.weather_days(conn, "2026-10-31", "2026-11-02")}
    # the 31st is two days old: final. Yesterday and today are refreshed.
    assert [rows[d]["final"] for d in sorted(rows)] == [1, 0, 0]
    # today's figures only count the hours so far (no forecast): 15 hours 00:00..14:00 of the canned temperatures
    assert rows["2026-11-02"]["temp_max_c"] == 10.0 + 14 / 2 and rows["2026-11-02"]["precip_mm"] == 0.0
    assert rows["2026-11-01"]["precip_mm"] == 24.0 and rows["2026-11-01"]["code"] == 61
    assert db.weather_hours(conn, "2026-11-02", "2026-11-02")[-1]["hour"] == "2026-11-02T14:00"
    api.clear()
    done = weather.refresh(conn, datetime(2026, 11, 2, 15, 30))
    assert done["days"] == 2 and api[0][1]["past_days"] == 1
    assert conn.execute("SELECT COUNT(*) FROM weather_days").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM weather_hours").fetchone()[0] == 24 * 2 + 16


def test_day_becomes_final_two_days_on(conn, add_visit, api):
    add_visit(captured_at="2026-11-01T09:00:00")
    weather.refresh(conn, NOW)
    assert db.weather_missing_days(conn, "2026-11-01", "2026-11-02") == ["2026-11-01", "2026-11-02"]
    api.clear()
    weather.refresh(conn, datetime(2026, 11, 3, 8, 0))  # the 1st is now two days old
    assert api[0][1]["past_days"] == 2
    assert db.weather_missing_days(conn, "2026-11-01", "2026-11-03") == ["2026-11-02", "2026-11-03"]
    api.clear()
    weather.refresh(conn, datetime(2026, 11, 3, 9, 0))
    assert api[0][1]["past_days"] == 1  # the 1st is never fetched again


def test_old_days_come_from_the_archive_in_one_call(conn, add_visit, monkeypatch):
    calls = []

    def fake(url, **params):
        calls.append((url, params))
        if url == weather.ARCHIVE_URL:
            return canned("2026-07-01", "2026-07-03")
        return canned("2026-11-01", "2026-11-02")

    monkeypatch.setattr(weather, "_get", fake)
    add_visit(captured_at="2026-07-01T09:00:00")
    # make the gap small for the test: only July 1..3 are "old" by pretending the forecast window is 5 days
    monkeypatch.setattr(weather, "FORECAST_DAYS", 3)
    monkeypatch.setattr(weather, "FINAL_AFTER", 2)
    weather.refresh(conn, datetime(2026, 7, 8, 10, 0))
    urls = [c[0] for c in calls]
    assert urls[0] == weather.ARCHIVE_URL and calls[0][1]["start_date"] == "2026-07-01"
    assert calls[0][1]["end_date"] == "2026-07-04"
    assert weather.FORECAST_URL in urls


def test_nothing_without_coordinates_visits_or_switch(conn, add_visit, api, monkeypatch):
    assert weather.refresh(conn, NOW) is None  # no visits
    add_visit(captured_at="2026-10-31T09:00:00")
    monkeypatch.setattr(config, "SITE_LAT", None)
    assert weather.refresh(conn, NOW) is None
    monkeypatch.setattr(config, "SITE_LAT", 1.5)
    monkeypatch.setattr(config, "WEATHER", False)
    assert weather.refresh(conn, NOW) is None
    assert api == [] and weather.start_background() is None
    assert conn.execute("SELECT COUNT(*) FROM weather_days").fetchone()[0] == 0


def test_network_error_does_not_raise_and_waits_an_hour(conn, add_visit, monkeypatch):
    add_visit(captured_at="2026-10-31T09:00:00")
    calls = []

    def boom(url, **params):
        calls.append(url)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(weather, "_get", boom)
    assert weather.refresh(conn, NOW) is None
    assert weather.refresh(conn, NOW) is None  # inside the wait: no second request
    assert len(calls) == 1
    monkeypatch.setattr(weather, "_failed_at", weather._failed_at - weather.RETRY_AFTER - 1)
    weather.refresh(conn, NOW)
    assert len(calls) == 2


def test_bad_json_does_not_raise(conn, add_visit, monkeypatch):
    add_visit(captured_at="2026-10-31T09:00:00")
    monkeypatch.setattr(weather, "_get", lambda url, **p: {"hourly": {"time": "nonsense"}, "daily": 5})
    assert weather.refresh(conn, NOW) is None


def test_days_option_refetches_final_days(conn, add_visit, api):
    add_visit(captured_at="2026-10-01T09:00:00")
    weather.refresh(conn, NOW, days=1)
    assert api[0][1]["past_days"] == 0
    assert db.weather_days(conn, "2026-10-01", "2026-11-02")[0]["day"] == "2026-11-02"


def test_old_database_gets_the_tables(tmp_path):
    with db.session(tmp_path / "old.db") as c:
        c.execute("CREATE TABLE visits (id INTEGER PRIMARY KEY, captured_at TEXT)")
        db.init_db(c)
        assert c.execute("SELECT COUNT(*) FROM weather_hours").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM weather_days").fetchone()[0] == 0
