"""Component tests of birdbrain/watch.py (stay): one real local dependency, a temp database or files on disk."""

from datetime import timedelta

import cv2

from helpers.clock import FROZEN_NOW
from helpers.stay import H, STILL, W, moving_at, open_stay, scene, with_bird
from birdbrain import stay, watch


def test_a_crow_walking_around_for_minutes_is_one_stay():
    st, base = open_stay(), scene()
    for t in range(0, 240):  # four minutes, one frame a second, wandering
        x, y = 100 + t, 150 + (t % 40)
        assert st.update(with_bird(base, x, y), moving_at(x, y), False, float(t)) is None
    for t in range(240, 240 + 29):
        assert st.update(base, STILL, False, float(t)) is None
    assert st.update(base, STILL, False, 270.0) == stay.GONE
    assert st.moved == 239.0


def test_watcher_holds_triggers_during_a_stay_and_saves_the_end(monkeypatch, conn):
    w = watch.Watcher()
    base = scene()
    w.detector.ring = None
    w.detector.box = None
    w.detector.ground_zone = ()
    vid = watch.db.insert_visit(conn, captured_at="2026-10-28T09:00:00", image_path="a.jpg", thumb_path="a_t.jpg",
                                visitor="bird", species="American Crow", confidence="high", model="claude-haiku")
    w.empty = base
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: (setattr(watch, "last_visit_id", vid), "bird")[1])
    w._work(0.0, probe=False, trigger="rim")
    assert w.stay_pending[0] == vid

    def feed(frame, t, moving=False):
        w.detector.full = frame
        w.detector.full_diff = (moving_at(300, 200) // 255 * 80) if moving else STILL
        return w.follow_stay(t)

    assert feed(with_bird(base, 300, 200), 1000.0, moving=True) is True
    assert feed(with_bird(base, 300, 200), 1010.0, moving=True) is True
    assert feed(base, 1030.0) is True
    assert feed(base, 1040.0) is False  # gone 30 s after last seen
    row = watch.db.get_visit(conn, vid)
    # The times fed in are monotonic; the stored end is the wall clock, 30 s ago (when it was last seen moving).
    assert row["ended_at"] == (FROZEN_NOW - timedelta(seconds=30)).isoformat()
    assert watch.json.loads(row["raw_json"])["stay_end"] == "gone"
    assert w.cooldown_until >= 1040.0 + stay.config.STAY_REARM


def test_false_triggers_never_open_a_stay(monkeypatch, conn):
    w = watch.Watcher()
    w.empty = scene()
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: "none")
    w._work(0.0, probe=False, trigger="rim")
    assert w.stay_pending is None


def test_the_watcher_keeps_the_scene_for_counting(monkeypatch, conn):
    w = watch.Watcher()
    w.detector.score(cv2.cvtColor(scene(), cv2.COLOR_GRAY2BGR))
    got = {}
    monkeypatch.setattr(watch, "calls_today", lambda ground=False: 0)
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: got.update(k) or "none")
    w._maybe_trigger(0.1)
    for _ in range(50):
        if not w.busy.is_set():
            break
        watch.time.sleep(0.01)
    assert got["scene"].empty.shape == (H, W) and got["scene"].mask.shape == (H, W)
    assert got["scene"].empty_color.shape == (H, W, 3)  # the colour copy, for counting
