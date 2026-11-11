"""Unit tests of birdbrain/presence: pure functions, no files, database or network."""

import numpy as np
import pytest

from helpers.presence import ABOVE, CROP, INSIDE
from helpers.presence_view import NARROW_CROP, RIM_LEFT, live, need
from birdbrain import presence


def test_coverage_of_a_circle_in_the_crop_box():
    assert presence.coverage(INSIDE, CROP) == 1.0
    assert presence.coverage(ABOVE, CROP) == 0.0
    half = presence.coverage({"x": 0.25, "y": 0.5, "r": 0.05}, CROP)  # centred on the box's left edge
    assert 0.45 < half < 0.55
    # r is a fraction of the width: on a 16:9 frame the circle reaches 16/9 as far up and down, so one just above
    # the box still dips into it
    assert 0 < presence.coverage({"x": 0.5, "y": 0.23, "r": 0.02}, CROP) < 0.2


def test_one_outlier_visitor_cant_drag_the_threshold_to_zero():
    scores = np.array([0.0006, 0.2, 0.3, 0.5, 0.9, 0.95] + [0.97] * 10)
    birds = np.array(["bird/day"] * 16, object)
    threshold, low, ignored = presence.threshold_from(scores, birds)
    assert (ignored, low) == (0, 1) and threshold == pytest.approx(0.8 * 0.2)
    # A first-of-its-kind visitor (the first possum at night) is never set aside: it lowers the threshold.
    first = birds.copy()
    first[0] = "possum/night"
    assert presence.threshold_from(scores, first)[2] is None
    assert presence.threshold_from(scores, first)[0] == pytest.approx(0.8 * 0.0006)
    # Nor is one that is merely the lowest of a smooth spread.
    assert presence.threshold_from(np.array([0.05] + [0.2] * 15), birds)[2] is None
    assert presence.threshold_from(scores)[2] is None  # no groups: the plain minimum


def test_a_visitor_the_check_cannot_see_is_set_aside_even_with_a_small_gap():
    # 2026-11-10: a squirrel in the shrub scored 0.005 against 0.02 next (gap 1.4 < OUTLIER_GAP) and dragged the
    # threshold to 0.004, so the check dropped 61% of empty frames instead of 85%.
    scores = np.array([0.005, 0.02, 0.025, 0.07, 0.1] + [0.9] * 12)
    birds = np.array(["squirrel/day"] * 17, object)
    threshold, low, ignored = presence.threshold_from(scores, birds)
    assert (ignored, low) == (0, 1) and threshold == pytest.approx(0.8 * 0.02)
    first = birds.copy()
    first[0] = "possum/night"  # the first of its kind still lowers the threshold
    assert presence.threshold_from(scores, first)[2] is None
    # two unseen ones are both skipped
    two = np.array([0.0004, 0.005, 0.02, 0.07] + [0.9] * 13)
    t2, low2, ign2 = presence.threshold_from(two, np.array(["bird/day"] * 17, object))
    assert (ign2, low2) == (0, 2) and t2 == pytest.approx(0.8 * 0.02)


def test_the_cli_refuses_redo_without_backfill(monkeypatch, capsys):
    monkeypatch.setattr(presence, "backfill", lambda *a, **k: pytest.fail("backfill ran"))
    monkeypatch.setattr("sys.argv", ["presence", "--redo"])
    with pytest.raises(SystemExit) as e:
        presence.main()
    assert e.value.code == 2 and "--redo goes with --backfill" in capsys.readouterr().err


def test_coverage_against_the_view_box_meets_the_minimum_for_the_rim_circle():
    box = need("presence_view_box")(live(), "rim")
    assert presence.coverage(RIM_LEFT, box) >= presence.MIN_COVERAGE
    assert presence.coverage(RIM_LEFT, NARROW_CROP) < presence.MIN_COVERAGE  # the old box missed it
