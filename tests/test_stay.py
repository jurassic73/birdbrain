from datetime import timedelta

import cv2
import numpy as np
from helpers.clock import FROZEN_NOW

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


def test_long_stays_are_capped(monkeypatch):
    monkeypatch.setattr(stay.config, "STAY_MAX", 20)  # the cap is read at each update; 20 frames instead of 900
    st, base = open_stay(), scene()
    t = 0
    while (r := st.update(with_bird(base, 100 + t % 50, 100), moving_at(100 + t % 50, 100), False, float(t))) is None:
        t += 1
    assert r == stay.LONG and t == stay.config.STAY_MAX == 20


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


def test_stayed_label():
    assert webapp.stayed("2026-10-28T09:00:00", "2026-10-28T09:03:40") == "stayed 4 min"
    assert webapp.stayed("2026-10-28T09:00:00", "2026-10-28T09:00:40") == ""
    assert webapp.stayed("2026-10-28T09:00:00", None) == ""


def main_photo(small: np.ndarray) -> bytes:
    """A main-stream photo (1080p, color) of a substream-sized gray scene."""
    big = cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    return cv2.imencode(".jpg", cv2.cvtColor(big, cv2.COLOR_GRAY2BGR))[1].tobytes()


def test_counts_the_animals_in_the_photo():
    empty, everywhere = scene(), np.full((H, W), 255, np.uint8)
    assert stay.count_animals(main_photo(empty), empty, everywhere) == 0
    one = with_bird(empty, 200, 200)
    assert stay.count_animals(main_photo(one), empty, everywhere) == 1
    two = with_bird(one, 450, 120)
    assert stay.count_animals(main_photo(two), empty, everywhere) == 2
    # only the crop box counts
    assert stay.count_animals(main_photo(two), empty, everywhere, box=(0, 0.3, 0.5, 0.7)) == 1


def test_a_bird_and_its_splash_are_one_animal_and_water_alone_is_none():
    empty, everywhere = scene(), np.full((H, W), 255, np.uint8)
    splash = with_bird(with_bird(empty, 200, 200), 200 + 18 + 14 + 4, 200, r=14)  # 4 px apart
    assert stay.count_animals(main_photo(splash), empty, everywhere) == 1
    water = np.zeros((H, W), np.uint8)
    cv2.circle(water, (450, 250), 60, 255, -1)
    glare = with_bird(with_bird(empty, 200, 200), 450, 250, r=25)  # a reflection changed out on the water
    assert stay.count_animals(main_photo(glare), empty, everywhere, water=water) == 1


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


def test_the_colour_scene_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(stay.config, "COUNT_COLOR", False)
    d = watch.MotionDetector(None, 0.01)
    for _ in range(3):
        d.score(cv2.cvtColor(scene(), cv2.COLOR_GRAY2BGR))
    assert d.scene().empty_color is None and d.empty_scene().shape == (H, W)


RIM = (150, 190, 210)   # BGR: a pale beige rim, grey 191
FUR = (191, 191, 191)   # neutral grey fur: the same grey as the rim, 41 apart in blue


def colour_scene() -> np.ndarray:
    """A dark yard with a pale rim band across it, blurred like the watcher's frames."""
    img = np.full((H, W, 3), 70, np.uint8)
    img[170:200, :] = RIM
    return cv2.GaussianBlur(img, (21, 21), 0)


def with_grey_animal(base: np.ndarray, x: int, y0: int = 130, y1: int = 240, w: int = 40) -> np.ndarray:
    """A grey animal standing across the rim band: head above it, body on it, tail below it."""
    frame = base.copy()
    cv2.rectangle(frame, (x, y0), (x + w, y1), FUR, -1)
    return frame


def colour_photo(small: np.ndarray) -> bytes:
    big = cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    return cv2.imencode(".jpg", big)[1].tobytes()


def test_an_animal_on_the_pale_rim_is_one_animal_in_colour_but_two_in_grey():
    # Visits 830 and 1017: a jay's and a squirrel's grey middle matched the pale rim, so only the head and the tail
    # changed and the Pi saw a crowd. The same frames in colour show the whole animal.
    empty, everywhere = colour_scene(), np.full((H, W), 255, np.uint8)
    photo = colour_photo(with_grey_animal(empty, 300))
    assert stay.count_animals(photo, cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY), everywhere) == 2
    assert stay.count_animals(photo, empty, everywhere) == 1
    # two of them, apart, are still two
    two = colour_photo(with_grey_animal(with_grey_animal(empty, 300), 500))
    assert stay.count_animals(two, empty, everywhere) == 2
    # and nothing is nothing
    assert stay.count_animals(colour_photo(empty), empty, everywhere) == 0


def test_the_watcher_counts_against_the_colour_scene_when_it_has_one():
    empty, everywhere = colour_scene(), np.full((H, W), 255, np.uint8)
    photo = colour_photo(with_grey_animal(empty, 300))
    grey_only = watch.Scene(cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY), everywhere, None)
    assert watch.count_animals(grey_only, photo, None) == 2
    both = watch.Scene(cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY), everywhere, None, empty)
    assert watch.count_animals(both, photo, None) == 1
