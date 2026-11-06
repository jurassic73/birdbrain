"""Unit tests of birdbrain/stay: pure functions, no files, database or network."""

import cv2
import numpy as np

from helpers.stay import H, STILL, W, moving_at, open_stay, scene, with_bird
from birdbrain import stay, watch


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


def test_detail_lists_where_each_counted_animal_was():
    empty, everywhere = scene(), np.full((H, W), 255, np.uint8)
    two = with_bird(with_bird(empty, 200, 200), 450, 120)
    boxes = []
    assert stay.count_animals(main_photo(two), empty, everywhere, detail=boxes) == 2
    assert len(boxes) == 2 and all(len(b) == 4 and all(0 <= v <= 1 for v in b) for b in boxes)
    assert sorted(round(b[0], 1) for b in boxes) == [0.3, 0.7]  # the birds at x 200 and 450 of 640, near their centres


def test_the_watcher_counts_against_the_colour_scene_when_it_has_one():
    empty, everywhere = colour_scene(), np.full((H, W), 255, np.uint8)
    photo = colour_photo(with_grey_animal(empty, 300))
    grey_only = watch.Scene(cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY), everywhere, None)
    assert watch.count_animals(grey_only, photo, None) == 2
    both = watch.Scene(cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY), everywhere, None, empty)
    assert watch.count_animals(both, photo, None) == 1
