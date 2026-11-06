"""Unit tests of birdbrain/sun: pure functions, no files, database or network."""

from datetime import date

from helpers.autocal import vancouver
from birdbrain import sun


def test_sun_times_follow_the_seasons(vancouver):
    assert sun.sunrise(date(2026, 10, 27)).strftime("%H:%M") in ("07:41", "07:42", "07:43")  # published: 07:42
    assert sun.sunset(date(2026, 10, 27)).strftime("%H:%M") in ("18:05", "18:06", "18:07")  # published: 18:06
    assert sun.sunset(date(2026, 6, 21)).hour == 21 and sun.sunset(date(2026, 12, 21)).hour == 16
