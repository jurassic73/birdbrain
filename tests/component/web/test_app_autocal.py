"""Component tests of the web app through TestClient on a temp database: autocal."""

from datetime import timedelta

import pytest

from helpers.autocal import RING, Z, env_zones, run_at, scene, vancouver
from birdbrain import autocal, config, db, watch, zones
from birdbrain.web import app as webapp


def test_settings_shows_the_zones_section_with_its_schedule(client):
    html = client.get("/settings").text
    assert 'id="zones"' in html and "Checked daily at 10:30 and 14:30" in html
    assert "Recalibrate now" in html and "No recalibrations yet" in html


def test_recalibrate_now_shows_progress_that_polls_until_done(client, conn, now):
    assert client.post("/settings/zones/now", follow_redirects=False).status_code == 303
    panel = client.get("/settings/zones").text
    assert "Looking at the bath" in panel and 'id="recal-timer"' in panel and "hx-trigger" not in panel
    assert client.get("/settings/zones/status").json()["pending"] is True
    run_at(conn, (now + timedelta(seconds=1)).isoformat())
    assert "recal-timer" not in client.get("/settings/zones").text
    assert client.get("/settings/zones/status").json()["pending"] is False


def test_saving_the_schedule(client, conn):
    client.post("/settings/zones", data={"time1": "11:00", "time2": "15:15"})
    assert {k: zones.schedule(conn)[k] for k in ("auto", "times")} == {"auto": False, "times": ["11:00", "15:15"]}
    assert client.post("/settings/zones", data={"auto": "1", "time1": "noon"}).status_code == 400


def test_undo_puts_the_zones_back(client, conn, env_zones):
    autocal.run([scene()], "scheduled")
    moved = zones.load(conn)
    assert moved != Z
    html = client.get("/settings").text
    cal_id = db.recent_calibrations(conn, 1)[0]["id"]
    assert f"/settings/zones/undo/{cal_id}" in html and "Use the .env zones" not in html
    assert 'data-tip="Moved:' in html
    client.post(f"/settings/zones/undo/{cal_id}")
    assert zones.load(conn) == Z and db.recent_calibrations(conn, 1)[0]["status"] == "undone"
    assert "/settings/zones/undo/" not in client.get("/settings").text   # nothing left to undo
    assert client.post("/settings/zones/reset").status_code in (404, 405)


def test_calibration_pictures_are_served_by_name_only(client, conn):
    autocal.run([scene()], "manual")
    name = db.recent_calibrations(conn, 1)[0]["image_path"]
    assert client.get(f"/calibration/{name}").headers["content-type"] == "image/jpeg"
    assert client.get("/calibration/reference.png").status_code == 404
    assert client.get("/calibration/..%2Fbirdbrain.db").status_code == 404


def test_about_draws_the_recalibrated_zones(client, conn, env_zones):
    zones.save(conn, zones.Zones(**{**Z.to_dict(), "spots": ((0.25, 0.5, 0.05, 0.05),)}))
    html = client.get("/about").text
    assert webapp.ellipse_path(0.25, 0.5, 0.05, 0.05) in html
    assert "as last adjusted automatically" in html and "Kept in place" in html


def test_about_zones_are_part_of_the_photo(client, conn, env_zones):
    html = client.get("/about").text
    svg = html[html.index('<div class="zones-frame">'):]
    svg = svg[:svg.index("</svg>")]
    assert '<image href="/static/about-zones.jpg' in svg and 'class="z-rim"' in svg  # one picture, zones inside it
    assert "<img" not in svg


def test_saving_sun_checkboxes(client, conn, vancouver):
    client.post("/settings/zones", data={"auto": "1", "time1": "", "sun1": "1", "time2": "15:15"})
    assert zones.schedule(conn)["times"] == ["sunrise", "15:15"]
    html = client.get("/settings").text
    assert "Use local sunrise" in html and "Checked daily at 1 h after sunrise" in html


def test_hand_adjusted_zones_are_saved_as_the_anchor_and_undo_drops_it(client, conn, env_zones):
    spot = [0.42, 0.455, 0.05, 0.06]
    ground = [[0.0, 0.65], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]]
    r = client.post("/settings/zones/manual", json={"ring": list(RING), "spots": [spot], "crop": [0.2, 0.2, 0.6, 0.6],
                                                     "ground": ground})
    assert r.status_code == 200
    z = zones.load(conn)
    assert z.spots == (tuple(spot),) and z.crop == (0.2, 0.2, 0.6, 0.6)
    assert z.ground_crop == (0.0, 0.65, 1.0, 0.35)   # it was the polygon's box, so it follows the polygon
    assert zones.manual(conn) == z
    assert 'data-which="zones"' in client.get("/settings").text   # the shape editor is on the photo
    autocal.run([scene()], "scheduled")             # an automatic run keeps the anchor
    assert zones.manual(conn) == z
    hand = next(c for c in db.recent_calibrations(conn) if c["reason"] == "hand")
    client.post(f"/settings/zones/undo/{hand['id']}")
    assert zones.manual(conn) is None
    assert client.post("/settings/zones/manual", json={"spots": [[0.1, 0.2]]}).status_code == 400


def test_zone_shapes_for_the_editor_round_trip_and_save(client, conn, env_zones):
    z = zones.load(conn)
    objs = zones.objects(z)
    roles = [o["role"] for o in objs]
    assert roles[0] == "ring" and "crop" in roles
    back = zones.edited(z, zones.from_objects(objs))
    icy = lambda r: r[6] if len(r) > 6 else r[1]  # noqa: E731
    assert back.ring[:6] == z.ring[:6] and icy(back.ring) == icy(z.ring) and back.spots == z.spots and back.crop == pytest.approx(z.crop)
    assert roles[1] == "water"  # the rim and the water are separate shapes
    # the editor moved the water down, drew a spot as points and a crooked crop box
    water = next(o for o in objs if o["role"] == "water")
    water["ellipse"][1] += 0.02
    objs = [o for o in objs if o["role"] != "spot"] + [
        {"role": "spot", "points": [[0.4, 0.4], [0.5, 0.4], [0.45, 0.5]]}]
    crop = next(o for o in objs if o["role"] == "crop")
    crop["points"] = [[0.1, 0.2], [0.7, 0.25], [0.65, 0.8], [0.12, 0.75]]
    assert client.post("/settings/zones/manual", json={"objects": objs}).status_code == 200
    saved = zones.load(conn)
    assert len(saved.ring) == 7 and saved.ring[6] == pytest.approx(water["ellipse"][1], abs=1e-4)
    assert saved.spots == ((0.45, 0.45, 0.05, 0.05),)
    assert saved.crop == pytest.approx((0.1, 0.2, 0.6, 0.6))
    assert client.post("/settings/zones/manual", json={"objects": [{"role": "moon"}]}).status_code == 400


def test_turned_and_off_center_shapes_save_and_mask(client, conn, env_zones):
    objs = zones.objects(zones.load(conn))
    rim, water, spot = objs[0], objs[1], objs[2]
    rim["ellipse"][4] = 20
    water["ellipse"][0] += 0.01
    water["ellipse"][4] = -95  # past -90 wraps to the same ellipse turned 85
    spot["ellipse"][4] = 30
    assert client.post("/settings/zones/manual", json={"objects": objs}).status_code == 200
    z = zones.load(conn)
    outer, inner = config.ring_ellipses(z.ring)
    assert outer[4] == 20 and inner[4] == 85 and inner[0] == pytest.approx(0.51)
    assert z.spots[0][4] == 30
    shifted = z.shifted(0.01, 0)
    assert config.ring_ellipses(shifted.ring)[1][0] == pytest.approx(0.52) and shifted.spots[0][4] == 30
    # a turned rim masks a turned band: the old axis-aligned mask differs
    turned = watch.motion_mask((360, 640), z.ring, ())
    flat = watch.motion_mask((360, 640), config.make_ring(outer[:4], inner[:4]), ())
    assert (turned != flat).sum() > 1000
    # back to level, a ring is saved in its short form again
    assert config.make_ring((0.5, 0.5, 0.2, 0.2, 0), (0.5, 0.48, 0.1, 0.1, 0)) == (0.5, 0.5, 0.2, 0.2, 0.1, 0.1, 0.48)


def test_claudes_notes_shows_the_settings_zones_read_only(client, conn, env_zones):
    # one source of truth: the Notes page draws the zones from Settings and links there, with no editor of its own
    page = client.get("/notes").text
    names = [o["name"] for o in zones.objects(zones.load(conn))]
    assert all(n in page for n in names) and 'href="/settings#zones"' in page
    assert "data-save" not in page and "/notes/scene" not in page and "Save and outline" not in page
    assert client.post("/notes/scene/outline").status_code in (404, 405)
