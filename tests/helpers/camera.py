"""Shared helpers for the camera tests (used by more than one test module)."""
import pytest

from birdbrain import camera, config


@pytest.fixture
def cam(monkeypatch):
    """A fake Thingino: remembers what was posted and answers like the real CGIs."""
    state = {"image": dict(camera.PICTURE_DEFAULTS, hflip=False),
             "daynight": {"enabled": True, "night_threshold": 25, "day_threshold": 50,
                          "transition_delay_s": 5, "force_mode": "",
                          "controls": {"ir850": True, "ir940": True},
                          "schedule": {"enabled": False, "start_at": "06:00", "stop_at": "22:00"}},
             "posts": []}

    def call(name, body=None):
        if body is not None:
            state["posts"].append((name, body))
        if name == "json-prudynt-config.cgi":
            return {"image": dict(state["image"])}
        if name == "json-prudynt-save.cgi":
            state["image"].update(body["image"])
            return {"status": "ok"}
        if name == "json-config-daynight.cgi":
            for k, v in (body or {}).get("daynight", {}).items():  # the camera merges nested sections
                if isinstance(v, dict):
                    state["daynight"].setdefault(k, {}).update(v)
                else:
                    state["daynight"][k] = v
            return dict(state["daynight"])
        if name == "json-daynight-sensors.cgi":
            return {"mode": "day", "daynight_brightness": 61, "time_now": 0}
        if name == "json-system-usage.cgi":
            return {"data": {"memory": {"total": 100, "free": 30, "cached": 10, "buffers": 10}}}
        raise camera.CameraError("unknown")

    monkeypatch.setattr(config, "CAM_API_KEY", "k")
    monkeypatch.setattr(camera, "_call", call)
    return state
