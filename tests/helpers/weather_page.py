"""Shared helpers for the weather page tests (used by more than one test module)."""
import json
from datetime import datetime

import pytest

from birdbrain import db


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
