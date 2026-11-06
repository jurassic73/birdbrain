"""Component tests of birdbrain/zones.py: one real local dependency, a temp database or files on disk."""

import json
from datetime import date, datetime, timedelta

import cv2
import pytest

from helpers.autocal import H, RING, ROCKS, SPOT, W, Z, env_zones, gray, run_at, scene, vancouver
from birdbrain import autocal, config, db, zones


@pytest.mark.smoke
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


def test_schedule_defaults_to_twice_a_day_and_saves(conn):
    assert {k: zones.schedule(conn)[k] for k in ("auto", "times")} == {"auto": True, "times": ["10:30", "14:30"]}
    zones.save_schedule(conn, False, ["15:00", "", "09:00"])
    assert {k: zones.schedule(conn)[k] for k in ("auto", "times")} == {"auto": False, "times": ["09:00", "15:00"]}
    with pytest.raises(ValueError):
        zones.save_schedule(conn, True, ["3pm"])


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


def test_manual_run_durations_give_the_timer_an_average(conn):
    from birdbrain import zones
    db.set_value(conn, "zones_request", "2026-10-31T14:55:00")
    zones.note_finished(conn, "manual", "2026-10-31T14:56:00")
    zones.note_finished(conn, "scheduled", "2026-10-31T14:59:00")   # not asked for: ignored
    db.set_value(conn, "zones_request", "2026-10-31T15:00:00")
    zones.note_finished(conn, "manual", "2026-10-31T15:01:30")
    assert zones.durations(conn) == [60, 90] and zones.status(conn)["average"] == 75


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
