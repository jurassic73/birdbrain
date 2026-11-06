"""Unit tests of birdbrain/alert: pure functions, no files, database or network."""

from datetime import time

from birdbrain import alert


def test_in_quiet_inside():
    assert alert._in_quiet(time(10, 0), "09:00", "11:00") is True


def test_in_quiet_before():
    assert alert._in_quiet(time(8, 0), "09:00", "11:00") is False


def test_in_quiet_at_end():
    # end is exclusive
    assert alert._in_quiet(time(11, 0), "09:00", "11:00") is False


def test_in_quiet_midnight_crossing_inside():
    assert alert._in_quiet(time(23, 30), "22:00", "07:00") is True


def test_in_quiet_midnight_crossing_early_morning():
    assert alert._in_quiet(time(3, 0), "22:00", "07:00") is True


def test_in_quiet_midnight_crossing_outside():
    assert alert._in_quiet(time(12, 0), "22:00", "07:00") is False


def test_in_quiet_bad_values():
    assert alert._in_quiet(time(10, 0), "bad", "11:00") is False
    assert alert._in_quiet(time(10, 0), "09:00", "") is False
