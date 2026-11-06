"""Unit tests of birdbrain/weather: pure functions, no files, database or network."""

import httpx
import pytest

from helpers.weather import canned, located  # noqa: F401  (autouse fixture)
from birdbrain import weather


def test_parse_rows_and_mean_cloud():
    days, hours = weather.parse(canned("2026-11-01", "2026-11-01"), "stamp")
    assert len(hours) == 24 and len(days) == 1
    h = hours[3]
    assert h["hour"] == "2026-11-01T03:00" and h["temp_c"] == 11.5 and h["wind_kmh"] == 7.5
    assert h["code"] == 3 and h["is_day"] == 0 and h["fetched_at"] == "stamp"
    d = days[0]
    assert d["day"] == "2026-11-01" and d["temp_max_c"] == 15.0 and d["temp_min_c"] == 5.0
    assert d["cloud_pct"] == 75.0 and d["sunrise"] == "2026-11-01T07:05" and d["sunset"] == "2026-11-01T16:55"


def test_parse_skips_empty_hours_and_tolerates_missing_sections():
    data = canned("2026-11-01", "2026-11-01")
    for k in data["hourly"]:
        if k != "time":
            data["hourly"][k][5:] = [None] * 19
    days, hours = weather.parse(data, "s")
    assert len(hours) == 5
    assert weather.parse({}, "s") == ([], [])


def test_http_error_status_is_a_failure(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(429, json={"reason": "limit"}))
    monkeypatch.setattr(weather.httpx, "get", lambda url, **kw: httpx.Client(transport=transport).get(url, **kw))
    with pytest.raises(httpx.HTTPStatusError):
        weather._get(weather.FORECAST_URL, latitude=1)


def test_describe():
    assert weather.describe(0) == ("clear", "Clear sky")
    assert weather.describe(2)[0] == "partly" and weather.describe(3)[0] == "cloudy"
    assert weather.describe(45)[0] == "fog" and weather.describe(53)[0] == "drizzle"
    assert weather.describe(63) == ("rain", "Rain") and weather.describe(82)[0] == "rain"
    assert weather.describe(75)[0] == "snow" and weather.describe(95)[0] == "storm"
    assert weather.describe(None) == ("cloudy", "Unknown") and weather.describe(1234)[0] == "cloudy"
    kinds = {weather.describe(c)[0] for c in weather._CODES}
    assert kinds == {"clear", "partly", "cloudy", "fog", "drizzle", "rain", "snow", "storm"}
