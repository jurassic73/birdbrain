"""Unit tests of birdbrain/autocal: pure functions, no files, database or network."""

import cv2
import numpy as np
import pytest

from helpers.autocal import H, RING, ROCKS, W, Z, gray, scene
from birdbrain import autocal, watch, zones


def test_a_dark_leaf_far_from_the_spot_is_ignored():
    out = autocal.recalibrate(scene(leaf=(0.6, 0.55)), Z, gray(scene()))
    assert "Ignored 1 other dark shape" in out.summary
    assert abs(out.zones.spots[0][0] - ROCKS[0]) < 0.012


def test_no_rocks_leaves_the_spot_alone():
    out = autocal.recalibrate(scene(rocks=None), Z, gray(scene(rocks=None)))
    assert out.status == "unchanged" and out.zones.spots == Z.spots and "left alone" in out.summary


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


def test_motion_detector_switches_to_new_zones():
    d = watch.MotionDetector(None, 0.02, Z.ring, Z.spots, Z.ground)
    d.score(scene())
    d.score(scene())
    assert d.mask is not None
    assert not d.use(Z)
    assert d.use(Z.shifted(0.02, 0)) and d.mask is None and d.ring[0] == pytest.approx(0.52)
    d.score(scene())
    assert d.mask is not None


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


def test_a_hand_set_spot_keeps_its_size_and_stays_while_the_rocks_are_inside_it():
    hand = zones.Zones(**{**Z.to_dict(), "spots": ((0.425, 0.46, 0.06, 0.07),)})
    out = autocal.recalibrate(scene(), hand, gray(scene()), anchor=hand)
    assert out.zones.spots == hand.spots and "where you put it" in out.summary
    out = autocal.recalibrate(scene(rocks=(0.56, 0.5)), hand, gray(scene()), anchor=hand)
    (cx, cy, rx, ry), = out.zones.spots
    assert abs(cx - 0.56) < 0.03 and (rx, ry) == (0.06, 0.07)   # followed the rocks, kept the hand size
