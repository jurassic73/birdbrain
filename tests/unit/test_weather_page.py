"""Unit tests of birdbrain/weather: pure functions, no files, database or network."""

from birdbrain import weather


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
