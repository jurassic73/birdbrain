import cv2
import numpy as np

from birdbrain import stay, watch
from birdbrain.web import app as webapp

H, W = 360, 640
STILL = np.zeros((H, W), np.uint8)


def scene() -> np.ndarray:
    rng = np.random.default_rng(1)
    return cv2.GaussianBlur(rng.integers(60, 140, (H, W), dtype=np.uint8), (21, 21), 0)


def with_bird(base: np.ndarray, x: int, y: int, r: int = 18) -> np.ndarray:
    frame = base.copy()
    cv2.circle(frame, (x, y), r, 10, -1)  # a dark crow
    return frame


def moving_at(x: int, y: int, r: int = 18) -> np.ndarray:
    m = STILL.copy()
    cv2.circle(m, (x, y), r, 255, -1)
    return m


def open_stay(now=0.0, night=False):
    return stay.Stay(7, scene(), np.full((H, W), 255, np.uint8), night, now)


def test_a_crow_walking_around_for_minutes_is_one_stay():
    st, base = open_stay(), scene()
    for t in range(0, 240):  # four minutes, one frame a second, wandering
        x, y = 100 + t, 150 + (t % 40)
        assert st.update(with_bird(base, x, y), moving_at(x, y), False, float(t)) is None
    for t in range(240, 240 + 29):
        assert st.update(base, STILL, False, float(t)) is None
    assert st.update(base, STILL, False, 270.0) == stay.GONE
    assert st.moved == 239.0


def test_a_crow_standing_still_for_a_while_still_counts():
    st, base = open_stay(), scene()
    crow = with_bird(base, 200, 200)
    st.update(crow, moving_at(200, 200), False, 0.0)
    for t in range(1, stay.config.STAY_STILL + 10):  # a whole minute without moving a feather
        assert st.update(crow, STILL, False, float(t)) is None


def test_leaves_it_moved_join_the_scene():
    """A crow flips a patch of leaves, then leaves: the patch stops counting after STAY_STILL, and movement
    elsewhere (wind in the bushes) doesn't keep it alive."""
    st, base = open_stay(), scene()
    leaves = with_bird(base, 300, 100, r=15)
    st.update(leaves, moving_at(300, 100), False, 0.0)
    ended = next(t for t in range(1, 400) if st.update(leaves, moving_at(600, 300), False, float(t)))
    assert ended == stay.config.STAY_STILL - 1 + stay.config.STAY_GAP
    assert st.present(leaves) is False  # absorbed into the empty scene


def test_small_specks_are_not_a_visitor():
    st, base = open_stay(), scene()
    frame = base.copy()
    for x in range(50, 600, 60):
        cv2.circle(frame, (x, 300), 2, 0, -1)  # raindrops
    assert st.present(frame) is False


def test_light_or_ir_change_closes_the_stay():
    st, base = open_stay(), scene()
    assert st.update(cv2.add(base, 60), STILL, False, 1.0) == stay.LIGHT
    assert open_stay().update(base, STILL, True, 1.0) == stay.LIGHT


def test_long_stays_are_capped():
    st, base = open_stay(), scene()
    t = 0
    while (r := st.update(with_bird(base, 100 + t % 50, 100), moving_at(100 + t % 50, 100), False, float(t))) is None:
        t += 1
    assert r == stay.LONG and t == stay.config.STAY_MAX


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
    assert row["ended_at"] == watch.datetime.fromtimestamp(1010.0).isoformat()
    assert watch.json.loads(row["raw_json"])["stay_end"] == "gone"
    assert w.cooldown_until >= 1040.0 + stay.config.STAY_REARM


def test_false_triggers_never_open_a_stay(monkeypatch, conn):
    w = watch.Watcher()
    w.empty = scene()
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: "none")
    w._work(0.0, probe=False, trigger="rim")
    assert w.stay_pending is None


def test_stayed_label():
    assert webapp.stayed("2026-10-28T09:00:00", "2026-10-28T09:03:40") == "stayed 4 min"
    assert webapp.stayed("2026-10-28T09:00:00", "2026-10-28T09:00:40") == ""
    assert webapp.stayed("2026-10-28T09:00:00", None) == ""
