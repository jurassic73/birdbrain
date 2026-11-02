"""The Weather & visits page: humidity and wind capture, the report math (db.weather_report) and the /weather route."""
import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from birdbrain import clock, config, db, weather
from birdbrain.web import app as webapp
from birdbrain.web import auth

NOW = datetime(2026, 11, 1, 12, 30)


def hour_row(hour: str, precip: float = 0.0, wind: float = 5.0, humidity: float = 85.0, temp: float = 10.0) -> dict:
    return {"hour": hour, "temp_c": temp, "precip_mm": precip, "rain_mm": precip, "cloud_pct": 100.0, "wind_kmh": wind,
            "humidity_pct": humidity, "code": 3, "is_day": 1 if "T07" <= hour[10:13] <= "T16" else 0, "fetched_at": "s"}


def day_row(day: str, precip: float) -> dict:
    return {"day": day, "temp_max_c": 12.0, "temp_min_c": 4.0, "precip_mm": precip, "precip_hours": 6.0, "cloud_pct": 90.0,
            "humidity_pct": 88.0, "wind_kmh": 9.0, "wind_max_kmh": 20.0, "code": 61 if precip else 3,
            "sunrise": f"{day}T07:00", "sunset": f"{day}T17:00", "final": 1, "fetched_at": "s"}


@pytest.fixture
def week(conn, add_visit):
    """Two days: Oct 31 rains from 9 to 15 (rows 10:00 to 15:00 carry the hour before), Nov 1 is dry and windy.
    The camera starts at 08:10 on Oct 31; now is Nov 1, 12:30."""
    hours = []
    for h in range(24):
        hours.append(hour_row(f"2026-10-31T{h:02d}:00", precip=1.0 if 10 <= h <= 15 else 0.0))
        hours.append(hour_row(f"2026-11-01T{h:02d}:00", wind=20.0, humidity=97.0, temp=3.0))
    db.save_weather(conn, [day_row("2026-10-31", 6.0), day_row("2026-11-01", 0.0)], hours)
    add_visit(captured_at="2026-10-31T08:10:00", visitor="none", species=None)               # the first photo
    add_visit(captured_at="2026-10-31T10:15:00")                                               # rainy hour 10
    add_visit(captured_at="2026-10-31T10:40:00",
              also=json.dumps([{"visitor": "bird", "species": "Bushtit", "count": 3}]))     # rainy hour 10
    add_visit(captured_at="2026-10-31T20:00:00", visitor="raccoon", species=None)            # after dark
    add_visit(captured_at="2026-11-01T08:05:00")
    add_visit(captured_at="2026-11-01T08:30:00")
    add_visit(captured_at="2026-11-01T09:00:00", visitor="squirrel", species="Eastern Gray Squirrel")
    add_visit(captured_at="2026-11-01T09:20:00", visitor="none", species=None)               # a false trigger
    conn.execute("INSERT INTO counters (day, name, n) VALUES ('2026-11-01', 'probe_empty', 2)")
    conn.commit()
    return conn


def test_humidity_is_parsed_and_day_means_computed():
    data = {"hourly": {"time": [f"2026-11-01T{h:02d}:00" for h in range(24)],
                       "temperature_2m": [5.0] * 24, "precipitation": [0.0] * 24, "rain": [0.0] * 24,
                       "cloud_cover": [50.0] * 24, "relative_humidity_2m": [80.0] * 12 + [100.0] * 12,
                       "wind_speed_10m": [2.0] * 7 + [10.0] * 10 + [30.0] + [2.0] * 6, "weather_code": [3] * 24,
                       "is_day": [0] * 7 + [1] * 10 + [0] * 7},
            "daily": {"time": ["2026-11-01"], "temperature_2m_max": [9.0], "temperature_2m_min": [2.0],
                      "precipitation_sum": [0.0], "precipitation_hours": [0.0], "weather_code": [3],
                      "sunrise": ["2026-11-01T07:00"], "sunset": ["2026-11-01T17:00"]}}
    days, hours = weather.parse(data, "s")
    assert hours[0]["humidity_pct"] == 80.0 and hours[23]["humidity_pct"] == 100.0
    d = days[0]
    assert d["humidity_pct"] == 90.0          # mean of every hour
    assert d["wind_kmh"] == 10.0              # mean of the daytime hours only
    assert d["wind_max_kmh"] == 30.0          # the windiest hour, day or night
    assert "relative_humidity_2m" in weather.HOURLY


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


def test_times_move_to_real_local_time_across_daylight_saving():
    """Open-Meteo sends one fixed offset (today's, after the clocks went back); hours before the change move an hour."""
    data = {"timezone": "America/Los_Angeles", "utc_offset_seconds": -8 * 3600,
            "hourly": {"time": ["2026-10-25T06:00", "2026-11-02T06:00"], "temperature_2m": [5.0, 5.0]},
            "daily": {"time": ["2026-10-25"], "temperature_2m_max": [9.0],
                      "sunrise": ["2026-10-25T06:41"], "sunset": ["2026-10-25T17:07"]}}
    days, hours = weather.parse(data, "s")
    assert [h["hour"] for h in hours] == ["2026-10-25T07:00", "2026-11-02T06:00"]
    assert days[0]["sunrise"] == "2026-10-25T07:41" and days[0]["sunset"] == "2026-10-25T18:07"
    assert weather.parse({**data, "timezone": None}, "s")[1][0]["hour"] == "2026-10-25T06:00"


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


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: NOW)
    with TestClient(webapp.app) as c:
        yield c


def test_page_with_weather(client, week):
    html = client.get("/weather").text
    assert "Weather &amp; visits" in html and "2 days since Oct 31" in html
    assert "so far" in html and "wx-scatter" in html and "Dark-eyed Junco" in html
    assert "% cloud cover" in html and "mph" in html and "88%" in html


def test_page_shows_na_for_days_without_humidity(client, week):
    week.execute("UPDATE weather_days SET humidity_pct = NULL, wind_kmh = NULL, wind_max_kmh = NULL")
    week.commit()
    html = client.get("/weather").text
    assert "n/a" in html


def test_page_without_weather(client, add_visit, monkeypatch):
    add_visit(captured_at="2026-11-01T08:05:00")
    monkeypatch.setattr(config, "SITE_LAT", None)
    assert "Weather isn't set up" in client.get("/weather").text
    monkeypatch.setattr(config, "SITE_LAT", 1.5)
    monkeypatch.setattr(config, "SITE_LON", 2.5)
    monkeypatch.setattr(config, "WEATHER", True)
    assert "No weather saved yet" in client.get("/weather").text


def test_page_with_no_visits(client):
    assert "No photos yet" in client.get("/weather").text


def test_read_only_visitors_see_it(client, week, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()
    page = client.get("/weather", follow_redirects=False)
    assert page.status_code == 200 and "Rain and visitors" in page.text


def test_stats_and_about_link_to_it(client, add_visit):
    add_visit()
    assert 'href="/weather"' in client.get("/stats").text
    assert 'href="/weather"' in client.get("/about").text
