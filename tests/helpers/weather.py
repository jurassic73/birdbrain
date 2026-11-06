"""Shared helpers for the weather tests (used by more than one test module)."""
from datetime import date, timedelta

import pytest

from birdbrain import config, weather


def canned(first: str, last: str, rain_on: str = "") -> dict:
    """An Open-Meteo-shaped response with 24 hourly rows per day from `first` through `last` (YYYY-MM-DD)."""
    a, b = date.fromisoformat(first), date.fromisoformat(last)
    days = [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]
    times = [f"{d}T{h:02d}:00" for d in days for h in range(24)]
    n = len(times)
    wet = [d == rain_on for d in days]
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [10.0 + (i % 24) / 2 for i in range(n)],
            "precipitation": [1.0 if t[:10] == rain_on else 0.0 for t in times],
            "rain": [1.0 if t[:10] == rain_on else 0.0 for t in times],
            "cloud_cover": [50.0 if i % 2 else 100.0 for i in range(n)],
            "wind_speed_10m": [7.5] * n,
            "weather_code": [3] * n,
            "is_day": [1 if 7 <= i % 24 <= 17 else 0 for i in range(n)],
        },
        "daily": {
            "time": days,
            "temperature_2m_max": [15.0] * len(days),
            "temperature_2m_min": [5.0] * len(days),
            "precipitation_sum": [24.0 if w else 0.0 for w in wet],
            "precipitation_hours": [24.0 if w else 0.0 for w in wet],
            "weather_code": [61 if w else 3 for w in wet],
            "sunrise": [f"{d}T07:05" for d in days],
            "sunset": [f"{d}T16:55" for d in days],
        },
    }


@pytest.fixture(autouse=True)
def located(monkeypatch):
    monkeypatch.setattr(config, "SITE_LAT", 1.5)
    monkeypatch.setattr(config, "SITE_LON", 2.5)
    monkeypatch.setattr(config, "WEATHER", True)
    monkeypatch.setattr(weather, "_failed_at", 0.0)
