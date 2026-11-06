import json
import os
import time
from datetime import date, datetime, timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from birdbrain import autocal, config, db, sun, watch, zones
from birdbrain.web import app as webapp

H, W = 360, 640
RING = (0.5, 0.5, 0.2, 0.25, 0.16, 0.2, 0.5)
ROCKS = (0.42, 0.45)  # where the rocks really are, as fractions
SPOT = (0.46, 0.48, 0.035, 0.045)  # the calibrated spot, a bit off the rocks


def scene(rocks=ROCKS, leaf=None, shift=(0, 0), seed=1) -> np.ndarray:
    """A daytime birdbath: textured ground, a pale rim, sky-bright water, a dark pile of rocks."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(40, 160, (H, W, 3), dtype=np.uint8), (0, 0), 3)
    cx, cy, orx, ory, irx, iry, icy = RING
    cv2.ellipse(img, (int(cx * W), int(cy * H)), (int(orx * W), int(ory * H)), 0, 0, 360, (190, 190, 190), -1)
    cv2.ellipse(img, (int(cx * W), int(icy * H)), (int(irx * W), int(iry * H)), 0, 0, 360, (215, 200, 185), -1)
    if rocks:
        for dx, dy in ((0, 0), (9, 3), (-6, 5)):
            cv2.ellipse(img, (int(rocks[0] * W) + dx, int(rocks[1] * H) + dy), (8, 5), 0, 0, 360, (55, 55, 60), -1)
    if leaf:
        cv2.ellipse(img, (int(leaf[0] * W), int(leaf[1] * H)), (7, 5), 0, 0, 360, (50, 45, 40), -1)
    if shift != (0, 0):
        img = cv2.warpAffine(img, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (W, H),
                             borderMode=cv2.BORDER_REPLICATE)
    return img


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


Z = zones.Zones(ring=RING, spots=(SPOT,), crop=(0.25, 0.2, 0.5, 0.6),
                ground=((0.0, 0.7), (1.0, 0.7), (1.0, 1.0), (0.0, 1.0)), ground_crop=(0.0, 0.7, 1.0, 0.3))


@pytest.fixture
def env_zones(monkeypatch):
    monkeypatch.setattr(config, "MOTION_RING", Z.ring)
    monkeypatch.setattr(config, "MOTION_SPOTS", Z.spots)
    monkeypatch.setattr(config, "MOTION_ROI", None)
    monkeypatch.setattr(config, "CROP_ROI", Z.crop)
    monkeypatch.setattr(config, "GROUND_ZONE", Z.ground)
    monkeypatch.setattr(config, "GROUND_CROP", Z.ground_crop)


# --- Zones in use ---

def test_zones_come_from_env_until_recalibrated(conn, env_zones):
    assert zones.load(conn) == Z
    moved = zones.Zones(**{**Z.to_dict(), "spots": ((0.42, 0.45, 0.03, 0.04),)})
    zones.save(conn, moved)
    assert zones.load(conn) == moved and zones.current() == moved


def test_a_hand_calibration_in_env_wins_over_the_automatic_one(conn, env_zones, monkeypatch):
    zones.save(conn, Z.shifted(0.01, 0))
    monkeypatch.setattr(config, "MOTION_SPOTS", ((0.3, 0.4, 0.03, 0.03),))
    assert zones.load(conn).spots == ((0.3, 0.4, 0.03, 0.03),) and zones.stored(conn) is None


def test_reset_goes_back_to_env(conn, env_zones):
    zones.save(conn, Z.shifted(0.01, 0))
    zones.reset(conn)
    assert zones.load(conn) == Z


def test_shifted_moves_every_zone_but_keeps_sizes():
    s = Z.shifted(0.01, -0.02)
    assert s.ring == (0.51, 0.48, 0.2, 0.25, 0.16, 0.2, 0.48)
    assert s.spots == ((0.47, 0.46, 0.035, 0.045),)
    assert s.crop == (0.26, 0.18, 0.5, 0.6)
    assert s.ground_crop == (0.01, 0.68, 0.99, 0.3)  # the full-width crop loses the sliver pushed off the right edge
    assert s.ground[0] == (0.01, 0.68)


def inside(z: zones.Zones) -> bool:
    """Every box, polygon corner and ellipse centre within the frame."""
    boxes = [b for b in (z.roi, z.crop, z.ground_crop) if b]
    ellipses = [*config.ring_ellipses(z.ring), *z.spots] if z.ring else list(z.spots)
    return (all(0 <= x and 0 <= y and x + w <= 1 and y + h <= 1 for x, y, w, h in boxes)
            and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in z.ground)
            and all(0 <= e[0] <= 1 and 0 <= e[1] <= 1 for e in ellipses))


def test_shifted_left_and_up_keeps_every_zone_inside_the_frame():
    # Z's ground zone and ground crop touch the left edge: a camera shift to the left used to push them to
    # negative x, which slices the frame from its far side and gives the watcher nothing to look at.
    s = Z.shifted(-0.05, -0.03)
    assert not inside(zones.Zones(ground_crop=(-0.05, 0.67, 1.0, 0.3)))  # what an unclamped shift gives
    assert inside(s)
    assert s.ground_crop == (0.0, 0.67, 0.95, 0.3) and s.ground[:2] == ((0.0, 0.67), (0.95, 0.67))
    assert s.crop == (0.2, 0.17, 0.5, 0.6) and s.ring[:2] == (0.45, 0.47)
    assert watch.roi(np.zeros((H, W, 3), np.uint8), s.ground_crop).shape[:2] == (108, 608)
    far = Z.shifted(-1.5, 0)  # pushed right off the frame: the shapes go, nothing is left half-wrapped
    assert far.ground == () and far.ground_crop is None and far.crop is None and inside(far)


# --- When it runs ---

def test_schedule_defaults_to_twice_a_day_and_saves(conn):
    assert {k: zones.schedule(conn)[k] for k in ("auto", "times")} == {"auto": True, "times": ["10:30", "14:30"]}
    zones.save_schedule(conn, False, ["15:00", "", "09:00"])
    assert {k: zones.schedule(conn)[k] for k in ("auto", "times")} == {"auto": False, "times": ["09:00", "15:00"]}
    with pytest.raises(ValueError):
        zones.save_schedule(conn, True, ["3pm"])


def run_at(conn, when: str, status="unchanged"):
    db.add_calibration(conn, ran_at=when, reason="scheduled", status=status, summary="")


def test_a_slot_is_due_once_within_its_window(conn):
    day = datetime(2026, 10, 26)
    assert zones.due(conn, day.replace(hour=10, minute=0)) is None
    assert zones.due(conn, day.replace(hour=10, minute=31)) == "scheduled"
    run_at(conn, "2026-10-26T10:33:00")
    assert zones.due(conn, day.replace(hour=10, minute=40)) is None
    assert zones.due(conn, day.replace(hour=13, minute=0)) is None   # missed slot? no, already ran
    assert zones.due(conn, day.replace(hour=14, minute=45)) == "scheduled"
    assert zones.due(conn, day.replace(hour=17, minute=0)) is None   # 14:30 slot is more than 2 h old


def test_skipped_runs_also_count_so_a_dark_slot_isnt_retried_every_minute(conn):
    run_at(conn, "2026-10-26T14:31:00", status="skipped")
    assert zones.due(conn, datetime(2026, 10, 26, 14, 50)) is None


def test_asking_runs_it_even_when_automatic_checks_are_off(conn, now):
    zones.save_schedule(conn, False, ["10:30"])
    assert zones.due(conn, datetime(2026, 10, 26, 10, 31)) is None
    zones.request(conn)
    assert zones.pending(conn) and zones.due(conn, now) == "manual"
    run_at(conn, (now + timedelta(seconds=1)).isoformat())  # the run finishes after the request, as it does for real
    assert not zones.pending(conn)


# --- Finding the rocks ---

def test_dark_clusters_find_the_rocks_but_not_the_bright_leaf_next_to_them():
    img = scene()
    cv2.ellipse(img, (int(0.47 * W), int(0.47 * H)), (14, 7), 0, 0, 360, (235, 235, 230), -1)  # pale floating leaf
    found = autocal.dark_clusters(gray(img), RING)
    assert len(found) == 1
    c = found[0]
    assert abs(c["cx"] - ROCKS[0] * W) < 6 and abs(c["cy"] - ROCKS[1] * H) < 6


def test_the_spot_moves_onto_the_rocks_and_then_stays_put():
    out = autocal.recalibrate(scene(), Z, gray(scene()))
    assert out.status == "moved" and "moved" in out.summary and "left and up" in out.summary
    (sx, sy, rx, ry), = out.zones.spots
    assert abs(sx - ROCKS[0]) < 0.012 and abs(sy - ROCKS[1]) < 0.02
    lo, hi = autocal.SPOT_SIZE_RANGE
    assert lo * SPOT[2] <= rx <= hi * SPOT[2] and lo * SPOT[3] <= ry <= hi * SPOT[3]
    again = autocal.recalibrate(scene(), out.zones, gray(scene()))
    assert again.status == "unchanged" and "still on the rocks" in again.summary


def test_a_dark_leaf_far_from_the_spot_is_ignored():
    out = autocal.recalibrate(scene(leaf=(0.6, 0.55)), Z, gray(scene()))
    assert "Ignored 1 other dark shape" in out.summary
    assert abs(out.zones.spots[0][0] - ROCKS[0]) < 0.012


def test_no_rocks_leaves_the_spot_alone():
    out = autocal.recalibrate(scene(rocks=None), Z, gray(scene(rocks=None)))
    assert out.status == "unchanged" and out.zones.spots == Z.spots and "left alone" in out.summary


# --- Noticing the camera move ---

def test_a_small_camera_move_moves_every_zone():
    on_rocks = Z.shifted(0, 0)
    on_rocks = zones.Zones(**{**on_rocks.to_dict(), "spots": ((ROCKS[0] + 0.004, ROCKS[1] + 0.008, 0.03, 0.04),)})
    out = autocal.recalibrate(scene(shift=(12, -6)), on_rocks, gray(scene()))
    assert out.camera_moved and "camera moved" in out.summary
    assert out.zones.ring[0] == pytest.approx(RING[0] + 12 / W, abs=0.003)
    assert out.zones.ring[1] == pytest.approx(RING[1] - 6 / H, abs=0.004)
    assert out.zones.ground[0][1] == pytest.approx(0.7 - 6 / H, abs=0.004)


def test_a_steady_camera_is_left_alone():
    out = autocal.recalibrate(scene(seed=1), Z, gray(scene(seed=1)))
    assert not out.camera_moved and "hasn't moved" in out.summary and out.zones.ring == Z.ring


def test_a_big_camera_move_is_reported_not_fixed():
    out = autocal.recalibrate(scene(shift=(80, 0)), Z, gray(scene()))
    assert out.status == "skipped" and out.zones == Z and "by hand" in out.summary


def test_the_first_run_saves_a_reference():
    out = autocal.recalibrate(scene(), Z, None)
    assert "reference" in out.summary and not out.camera_moved


# --- Running in the watcher ---

def test_run_saves_the_zones_a_picture_a_reference_and_a_record(conn, env_zones):
    zones.clear_reference()
    autocal.run([scene(seed=s) for s in (1, 2, 3)], "manual")
    assert zones.reference_path().exists()
    row = db.recent_calibrations(conn, 1)[0]
    assert row["status"] == "moved" and row["reason"] == "manual"
    assert (autocal.picture_dir() / row["image_path"]).exists()
    assert zones.Zones.from_dict(json.loads(row["before_json"])) == Z
    assert zones.load(conn).spots != Z.spots


@pytest.fixture
def inline_threads(monkeypatch):
    class Inline:
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(autocal.threading, "Thread", Inline)


def test_recalibrator_samples_quiet_frames_then_runs(conn, monkeypatch, inline_threads):
    monkeypatch.setattr(autocal, "SAMPLE_EVERY", 0)
    monkeypatch.setattr(autocal, "SAMPLES", 4)
    monkeypatch.setattr(autocal, "MIN_SAMPLES", 2)
    runs = []
    monkeypatch.setattr(autocal, "run", lambda frames, reason, night: runs.append((len(frames), reason, night)))
    zones.request(conn)
    r = autocal.Recalibrator()
    frame = scene()
    r.feed(frame, night=False, quiet=True)        # notices the request
    for quiet in (True, False, True, True, True):  # a busy frame isn't sampled
        r.feed(frame, night=False, quiet=quiet)
    assert runs == [(4, "manual", False)] and r.reason is None


def test_recalibrator_skips_when_the_light_switches_and_when_too_busy(conn, monkeypatch, inline_threads):
    zones.request(conn)
    r = autocal.Recalibrator()
    r.feed(scene(), night=False, quiet=True)
    r.feed(scene(), night=False, quiet=True)
    r.feed(scene(), night=True, quiet=True)  # dusk: the IR came on mid-sample
    assert "light changed" in db.recent_calibrations(conn, 1)[0]["summary"]

    conn.execute("UPDATE calibrations SET ran_at = '2026-01-01T00:00:00'")  # well before the next request
    conn.commit()
    zones.request(conn)
    r.next_check = 0
    r.feed(scene(), night=False, quiet=True)
    r.started -= autocal.MAX_WAIT + 1
    r.feed(scene(), night=False, quiet=False)
    row = db.recent_calibrations(conn, 1)[0]
    assert row["status"] == "skipped" and "Too much going on" in row["summary"]


def test_motion_detector_switches_to_new_zones():
    d = watch.MotionDetector(None, 0.02, Z.ring, Z.spots, Z.ground)
    d.score(scene())
    d.score(scene())
    assert d.mask is not None
    assert not d.use(Z)
    assert d.use(Z.shifted(0.02, 0)) and d.mask is None and d.ring[0] == pytest.approx(0.52)
    d.score(scene())
    assert d.mask is not None


# --- Settings page ---

@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


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


def test_manual_run_durations_give_the_timer_an_average(conn):
    from birdbrain import zones
    db.set_value(conn, "zones_request", "2026-10-31T14:55:00")
    zones.note_finished(conn, "manual", "2026-10-31T14:56:00")
    zones.note_finished(conn, "scheduled", "2026-10-31T14:59:00")   # not asked for: ignored
    db.set_value(conn, "zones_request", "2026-10-31T15:00:00")
    zones.note_finished(conn, "manual", "2026-10-31T15:01:30")
    assert zones.durations(conn) == [60, 90] and zones.status(conn)["average"] == 75


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


def test_about_numbers_the_zones_like_settings(client, conn, env_zones):
    html = client.get("/about").text
    svg = html[html.index('<div class="zones-frame">'):]
    svg = svg[:svg.index("</svg>")]
    assert svg.count('class="z-num"') >= 3  # rim, open water and the rest, each with its number on the photo
    assert '<span class="znum" aria-hidden="true">1</span>' in html  # and the same number in the legend


# --- Sunrise and sunset slots ---

@pytest.fixture
def vancouver(monkeypatch):
    old_tz = os.environ.get("TZ")
    monkeypatch.setattr(config, "SITE_LAT", 45.64)
    monkeypatch.setattr(config, "SITE_LON", -122.66)
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.delenv("TZ", raising=False)  # restore TZ (monkeypatch would only do it after we're done) ...
    if old_tz is not None:
        monkeypatch.setenv("TZ", old_tz)
    time.tzset()  # ... and tell libc, which only re-reads it here


def test_sun_times_follow_the_seasons(vancouver):
    assert sun.sunrise(date(2026, 10, 27)).strftime("%H:%M") in ("07:41", "07:42", "07:43")  # published: 07:42
    assert sun.sunset(date(2026, 10, 27)).strftime("%H:%M") in ("18:05", "18:06", "18:07")  # published: 18:06
    assert sun.sunset(date(2026, 6, 21)).hour == 21 and sun.sunset(date(2026, 12, 21)).hour == 16


def test_sun_slots_run_an_hour_inside_daylight(conn, vancouver):
    zones.save_schedule(conn, True, ["sunset", "sunrise"])
    s = zones.schedule(conn)
    assert s["times"] == ["sunrise", "sunset"] and s["labels"][1].startswith("1 h before sunset")
    rise = zones.slot_time("sunrise", date(2026, 10, 27))
    assert rise.strftime("%H:%M").startswith("08:4")
    assert zones.due(conn, rise - timedelta(minutes=1)) is None
    assert zones.due(conn, rise + timedelta(minutes=1)) == "scheduled"


def test_sun_slots_without_a_location_never_run(conn, monkeypatch):
    monkeypatch.setattr(config, "SITE_LAT", None)
    zones.save_schedule(conn, True, ["sunrise"])
    assert not zones.schedule(conn)["located"]
    assert all(zones.due(conn, datetime(2026, 10, 27, h)) is None for h in range(24))


def test_saving_sun_checkboxes(client, conn, vancouver):
    client.post("/settings/zones", data={"auto": "1", "time1": "", "sun1": "1", "time2": "15:15"})
    assert zones.schedule(conn)["times"] == ["sunrise", "15:15"]
    html = client.get("/settings").text
    assert "Use local sunrise" in html and "Checked daily at 1 h after sunrise" in html



def night_scene(shift=(0, 0)):
    """An IR birdbath: gray ground, a pale rim, black water, the rocks lit up."""
    img = scene(rocks=None, shift=(0, 0))
    g = gray(img)
    cx, cy, orx, ory, irx, iry, icy = RING
    cv2.ellipse(g, (int(cx * W), int(icy * H)), (int(irx * W), int(iry * H)), 0, 0, 360, 15, -1)
    for dx, dy in ((0, 0), (9, 3), (-6, 5)):
        cv2.ellipse(g, (int(ROCKS[0] * W) + dx, int(ROCKS[1] * H) + dy), (8, 5), 0, 0, 360, 150, -1)
    if shift != (0, 0):
        g = cv2.warpAffine(g, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def test_at_night_the_rocks_are_the_bright_shape_in_black_water():
    out = autocal.recalibrate(night_scene(), Z, None, night=True)
    sx, sy = out.zones.spots[0][:2]
    assert abs(sx - ROCKS[0]) < 0.02 and abs(sy - ROCKS[1]) < 0.03
    assert "night reference" in out.summary


def test_a_night_run_lines_up_against_the_day_reference_when_the_camera_moved():
    out = autocal.recalibrate(night_scene(shift=(12, 5)), Z, gray(scene()), night=True, cross=True)
    assert out.camera_moved and abs(out.zones.ring[0] - (RING[0] + 12 / W)) < 0.01


def test_a_night_run_against_the_day_reference_leaves_an_unmoved_camera_alone():
    out = autocal.recalibrate(night_scene(), Z, gray(scene()), night=True, cross=True)
    assert not out.camera_moved


# --- Adjust zones (by hand) ---


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


def test_a_hand_set_spot_keeps_its_size_and_stays_while_the_rocks_are_inside_it():
    hand = zones.Zones(**{**Z.to_dict(), "spots": ((0.425, 0.46, 0.06, 0.07),)})
    out = autocal.recalibrate(scene(), hand, gray(scene()), anchor=hand)
    assert out.zones.spots == hand.spots and "where you put it" in out.summary
    out = autocal.recalibrate(scene(rocks=(0.56, 0.5)), hand, gray(scene()), anchor=hand)
    (cx, cy, rx, ry), = out.zones.spots
    assert abs(cx - 0.56) < 0.03 and (rx, ry) == (0.06, 0.07)   # followed the rocks, kept the hand size


def test_claudes_notes_shows_the_settings_zones_read_only(client, conn, env_zones):
    # one source of truth: the Notes page draws the zones from Settings and links there, with no editor of its own
    page = client.get("/notes").text
    names = [o["name"] for o in zones.objects(zones.load(conn))]
    assert all(n in page for n in names) and 'href="/settings#zones"' in page
    assert "data-save" not in page and "/notes/scene" not in page and "Save and outline" not in page
    assert client.post("/notes/scene/outline").status_code in (404, 405)
