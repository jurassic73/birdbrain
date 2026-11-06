"""Circle from the motion: the Pi circles the animal on visits it named itself (birdbrain/motion_circle.py)."""

import json

import cv2
import numpy as np
import pytest

from helpers.motion_circle import RIM_TOP, ZONES, animal, circle, jpeg, near, scene
from birdbrain import config, db, motion_circle as mc


@pytest.mark.parametrize("shade", [25, 250])  # a dark bird by day, a bright animal in infrared: either direction
def test_circles_the_animal_on_the_rim(shade):
    empty = scene()
    c, info = circle(animal(empty, RIM_TOP, shade=shade), empty)
    assert c and near(c, RIM_TOP) and 0.015 < c["r"] < 0.05 and c["pi"] is True
    assert info["found"] and info["source"] == "stored" and info["seconds"] < 1


def test_same_form_as_claudes_circle():
    empty = scene()
    c, _ = circle(animal(empty, RIM_TOP), empty)
    assert set(c) == {"x", "y", "r", "pi"} and db.parse_marks(json.dumps([c])) == [c]


def test_a_sun_patch_is_only_light():
    """A patch of sun keeps the mulch's pattern, so it doesn't count; the bird elsewhere still does."""
    empty = scene()
    sunny = empty.copy()
    patch = sunny[270:445, 576:1344].astype(np.float32)  # the mulch above the bath, up to the rim
    sunny[270:445, 576:1344] = np.clip(patch * 1.6 + 20, 0, 255).astype(np.uint8)
    assert circle(sunny, empty)[0] is None
    bird = (0.64, 0.52)
    c, _ = circle(animal(sunny, bird), empty)
    assert c and near(c, bird)


def test_two_animals_of_a_size_give_no_circle():
    empty = scene()
    c, info = circle(animal(animal(empty, (0.4, 0.45)), (0.6, 0.45)), empty)
    assert c is None and info["why"] == "two alike" and not info["found"]


def test_a_much_smaller_second_change_doesnt_matter():
    empty = scene()
    c, _ = circle(animal(animal(empty, (0.4, 0.45), size=(90, 60)), (0.6, 0.45), size=(25, 18)), empty)
    assert c and near(c, (0.4, 0.45))


def test_too_big_a_change_is_the_light():
    empty = scene()
    c, info = circle(animal(empty, (0.5, 0.42), size=(400, 260)), empty)
    assert c is None and info["why"] in ("too big", "light")


def test_the_water_counts_only_for_one_compact_patch():
    empty = scene()
    sky = empty.copy()  # the reflection changed across the water (clouds): no animal
    outer, inner = config.ring_ellipses(ZONES.ring)
    config.draw_ellipse(sky, (inner[0] - 0.03, inner[1], inner[2] * 0.6, inner[3] * 0.6), 160)
    assert circle(sky, empty)[0] is None
    bather = (0.47, 0.56)
    c, info = circle(animal(empty, bather, size=(40, 30), shade=200), empty)
    assert c and near(c, bather) and info["where"] == "water"


def test_the_watchers_empty_scene_is_used_as_it_is():
    """The motion detector's background is already substream-sized, gray and blurred: it isn't blurred again."""
    empty = scene()
    background = cv2.GaussianBlur(cv2.resize(empty, mc.WORK, interpolation=cv2.INTER_AREA), (21, 21), 0)
    c, info = mc.live(jpeg(animal(empty, RIM_TOP)), background, ZONES, "rim", "2026-11-01T10:00:00", False)
    assert c and near(c, RIM_TOP) and info["source"] == "empty"


def test_fails_soft(monkeypatch):
    empty = scene()
    assert circle(empty, None)[1] == {"found": False, "why": "no empty view", "source": None}
    assert mc.circle_for(b"not a jpeg", empty, ZONES, "rim", "stored")[1]["why"] == "unreadable photo"
    monkeypatch.setattr(mc, "find", lambda *a: 1 / 0)
    c, info = circle(animal(empty, RIM_TOP), empty)
    assert c is None and info["why"] == "error"
