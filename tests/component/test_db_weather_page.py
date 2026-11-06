"""Component tests of birdbrain/db.py (weather page): one real local dependency, a temp database or files on disk."""

import pytest

from helpers.weather_page import NOW, day_row, hour_row, week
from birdbrain import db


def test_humidity_and_wind_are_saved(conn):
    db.save_weather(conn, [day_row("2026-11-01", 0.0)], [hour_row("2026-11-01T10:00", humidity=91.0)])
    assert db.weather_hours(conn, "2026-11-01", "2026-11-01")[0]["humidity_pct"] == 91.0
    d = db.weather_days(conn, "2026-11-01", "2026-11-01")[0]
    assert (d["humidity_pct"], d["wind_kmh"], d["wind_max_kmh"]) == (88.0, 9.0, 20.0)


def test_old_weather_tables_get_the_new_columns(tmp_path):
    with db.session(tmp_path / "old.db") as c:
        c.executescript("""
            CREATE TABLE weather_hours (hour TEXT PRIMARY KEY, temp_c REAL, precip_mm REAL, rain_mm REAL,
                cloud_pct REAL, wind_kmh REAL, code INTEGER, is_day INTEGER, fetched_at TEXT);
            CREATE TABLE weather_days (day TEXT PRIMARY KEY, temp_max_c REAL, temp_min_c REAL, precip_mm REAL,
                precip_hours REAL, cloud_pct REAL, code INTEGER, sunrise TEXT, sunset TEXT,
                final INTEGER NOT NULL DEFAULT 0, fetched_at TEXT);
            INSERT INTO weather_days (day, final) VALUES ('2026-10-25', 1);""")
        db.init_db(c)
        hours = {r["name"] for r in c.execute("PRAGMA table_info(weather_hours)")}
        days = {r["name"] for r in c.execute("PRAGMA table_info(weather_days)")}
        assert "humidity_pct" in hours and {"humidity_pct", "wind_kmh", "wind_max_kmh"} <= days
        assert c.execute("SELECT humidity_pct FROM weather_days").fetchone()[0] is None  # n/a until refetched
        db.init_db(c)  # twice is fine


def test_report_days(week):
    r = db.weather_report(week, now=NOW)
    oct31, nov1 = r["days"]
    assert oct31["visitors"] == 3 and oct31["day_visitors"] == 2           # the raccoon came after dark
    assert oct31["species"] == 3                                             # junco once (2 photos), bushtit, raccoon
    assert oct31["top"][0] == ("Dark-eyed Junco", 2)
    assert oct31["daylight"] == pytest.approx(8 + 50 / 60)                  # from the first photo at 08:10 to sunset
    assert oct31["false_alarms"] == 1 and oct31["first"] and not oct31["today"]
    assert nov1["today"] and nov1["daylight"] == pytest.approx(5.5)         # sunrise to now
    assert nov1["visitors"] == 3 and nov1["per_hour"] == pytest.approx(3 / 5.5)
    assert nov1["species"] == 2 and nov1["false_alarms"] == 3               # one stored, two probes
    assert r["max_rain"] == 6.0 and r["max_visitors"] == 3
    assert [p["partial"] for p in r["points"]] == [True, True]


def test_report_rainy_vs_dry_hours(week, monkeypatch):
    monkeypatch.setattr(db, "MIN_HOURS", 3)
    r = db.weather_report(week, now=NOW)
    # Whole daylight hours after 08:10 on Oct 31 (9 to 16) and up to 12:00 on Nov 1 (7 to 11): 13.
    assert r["hours"] == 13
    rain, dry = r["rain"]
    assert (rain["hours"], rain["visitors"]) == (6, 2) and rain["rate"] == pytest.approx(2 / 6)
    assert (dry["hours"], dry["visitors"]) == (7, 3) and dry["enough"]
    calm, windy = r["wind"]
    assert (windy["hours"], windy["visitors"]) == (5, 3) and (calm["hours"], calm["visitors"]) == (8, 2)
    assert windy["rate"] == pytest.approx(3 / 5)
    text = " ".join(db.weather_findings(r))
    assert "Rainy daylight hours had 0.3 visitors an hour (2 in 6 hours)" in text and "fewer than dry hours" in text
    assert "Windy hours" in text


def test_too_few_hours_are_not_a_finding(week):
    r = db.weather_report(week, now=NOW)
    assert not r["rain"][0]["enough"]
    findings = db.weather_findings(r)
    assert findings and findings[0].startswith("Not enough rainy daylight hours yet")
    assert not any("Windy" in f for f in findings)


def test_report_with_no_visits(conn):
    r = db.weather_report(conn, now=NOW)
    assert r["days"] == [] and not r["has_weather"]
