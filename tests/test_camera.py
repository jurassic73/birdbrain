import pytest
from fastapi.testclient import TestClient

from birdbrain import camera, config
from birdbrain.web import app as webapp


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


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


def test_clean_picture_clamps_and_drops_unknown_keys():
    assert camera.clean_picture({"brightness": "300", "backlight_compensation": "-2", "hflip": "1"}) == {
        "brightness": 255, "backlight_compensation": 0}


def test_clean_daynight_keeps_day_above_night():
    with pytest.raises(ValueError):
        camera.clean_daynight({"night_threshold": "50", "day_threshold": "40"})
    with pytest.raises(ValueError):
        camera.clean_daynight({"force_mode": "sideways"})
    assert camera.clean_daynight({"force_mode": "night"}) == {"force_mode": "night"}


def test_picture_change_is_logged_and_undone(conn, cam):
    camera.set_picture(conn, {"brightness": 150, "contrast": 128})
    assert cam["posts"] == [("json-prudynt-save.cgi", {"image": {"brightness": 150, "contrast": 128}})]
    (row,) = camera.history(conn)
    assert row["before"] == {"brightness": 128} and row["after"] == {"brightness": 150}  # unchanged contrast left out
    assert camera.undo(conn, row["id"])
    assert cam["image"]["brightness"] == 128 and camera.history(conn) == []


def test_daynight_is_posted_under_its_key(conn, cam):
    camera.set_daynight(conn, {"night_threshold": 20})
    assert cam["posts"][-1] == ("json-config-daynight.cgi", {"daynight": {"night_threshold": 20}})


def test_panel_without_key_explains_setup(client, monkeypatch):
    monkeypatch.setattr(config, "CAM_API_KEY", "")
    assert "CAM_API_KEY" in client.get("/settings/camera").text


def test_panel_shows_status_and_sliders(client, cam):
    html = client.get("/settings/camera").text
    assert "day mode" in html and "61" in html and 'name="brightness"' in html
    assert 'name="hflip"' not in html


def test_panel_when_camera_is_down(client, monkeypatch):
    monkeypatch.setattr(config, "CAM_API_KEY", "k")

    def down(name, body=None):
        raise camera.CameraError("camera didn't answer (ConnectTimeout)")
    monkeypatch.setattr(camera, "_call", down)
    assert "Couldn't reach the camera" in client.get("/settings/camera").text


def test_save_picture_route_and_defaults(client, cam):
    html = client.post("/settings/camera/picture", data={"brightness": "140"}).text
    assert "Saved on the camera" in html and "Undo" in html and cam["image"]["brightness"] == 140
    client.post("/settings/camera/picture", data={"brightness": "140", "defaults": "1"})
    assert cam["image"]["brightness"] == 128


def test_bad_daynight_shows_error(client, cam):
    html = client.post("/settings/camera/daynight", data={"night_threshold": "60", "day_threshold": "30"}).text
    assert "above the night level" in html and cam["posts"] == []


def test_nested_daynight_settings_are_flat_in_the_form_and_nested_on_the_camera(conn, cam):
    assert camera.daynight()["schedule.start_at"] == "06:00" and camera.daynight()["controls.ir850"] is True
    camera.set_daynight(conn, camera.clean_daynight({"controls.ir850": "0", "schedule.start_at": "07:30"}))
    assert cam["posts"][-1] == ("json-config-daynight.cgi",
                                {"daynight": {"controls": {"ir850": False}, "schedule": {"start_at": "07:30"}}})
    assert cam["daynight"]["schedule"]["stop_at"] == "22:00"  # the rest of the section is kept
    (row,) = camera.history(conn)
    assert camera.undo(conn, row["id"])
    assert cam["daynight"]["controls"]["ir850"] is True and cam["daynight"]["schedule"]["start_at"] == "06:00"


def test_toggles_and_times_are_checked():
    # a checked box sends the hidden 0 then 1; the last one wins
    assert camera.clean_daynight({"controls.ir940": "1"}) == {"controls.ir940": True}
    with pytest.raises(ValueError):
        camera.clean_daynight({"schedule.start_at": "25:00"})
    with pytest.raises(ValueError):
        camera.clean_picture({"core_wb_mode": "42"})
    assert camera.clean_picture({"anti_flicker": "1", "hue": "999"}) == {"anti_flicker": 1, "hue": 255}


def test_defaults_only_reset_the_open_tab(client, cam):
    cam["image"].update(brightness=200, saturation=40)
    html = client.post("/settings/camera/picture", data={"tab": "color", "defaults": "1"}).text
    assert cam["image"]["saturation"] == 128 and cam["image"]["brightness"] == 200
    assert 'data-cam-tab="color" aria-selected="true"' in html  # the panel comes back on the same tab


def test_panel_has_every_tab_and_the_live_view(client, cam):
    html = client.get("/settings/camera").text
    for name in ("Picture", "Light", "Color", "Day and night", "Changes"):
        assert f">{name}" in html
    for key in ("sinter_strength", "max_again", "core_wb_mode", "controls.ir940", "schedule.stop_at"):
        assert f'name="{key}"' in html
    assert 'id="cam-live-img"' in html


def test_settings_page_loads_camera_panel_lazily(client):
    assert 'hx-get="/settings/camera"' in client.get("/settings").text
