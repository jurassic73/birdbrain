"""Unit tests of birdbrain/camera: pure functions, no files, database or network."""

import time

import pytest

from birdbrain import camera


def test_clean_picture_clamps_and_drops_unknown_keys():
    assert camera.clean_picture({"brightness": "300", "backlight_compensation": "-2", "hflip": "1"}) == {
        "brightness": 255, "backlight_compensation": 0}


def test_clean_daynight_keeps_day_above_night():
    with pytest.raises(ValueError):
        camera.clean_daynight({"night_threshold": "50", "day_threshold": "40"})
    with pytest.raises(ValueError):
        camera.clean_daynight({"force_mode": "sideways"})
    assert camera.clean_daynight({"force_mode": "night"}) == {"force_mode": "night"}


def test_toggles_and_times_are_checked():
    # a checked box sends the hidden 0 then 1; the last one wins
    assert camera.clean_daynight({"controls.ir940": "1"}) == {"controls.ir940": True}
    with pytest.raises(ValueError):
        camera.clean_daynight({"schedule.start_at": "25:00"})
    with pytest.raises(ValueError):
        camera.clean_picture({"core_wb_mode": "42"})
    assert camera.clean_picture({"anti_flicker": "1", "hue": "999"}) == {"anti_flicker": 1, "hue": 255}


def test_camera_status_survives_string_time_now(monkeypatch):
    def call(name, *a, **kw):
        if name == "json-daynight-sensors.cgi":
            return {"mode": "day", "daynight_brightness": 5, "time_now": str(time.time() - 7)}
        raise camera.CameraError("x")
    monkeypatch.setattr(camera, "_call", call)
    assert abs(camera.status()["clock_off"] - -7) <= 1
    for bad in ("not a time", [1], {"a": 1}):
        monkeypatch.setattr(camera, "_call", lambda name, *a, _b=bad, **kw: (
            {"mode": "day", "time_now": _b} if name == "json-daynight-sensors.cgi" else (_ for _ in ()).throw(camera.CameraError("x"))))
        out = camera.status()
        assert "clock_off" not in out and out["mode"] == "day"
