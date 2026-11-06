"""Component tests of the web app through TestClient on a temp database: camera."""

from helpers.camera import cam
from birdbrain import camera, config


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
