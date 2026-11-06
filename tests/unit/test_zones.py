"""Unit tests of birdbrain/zones: pure functions, no files, database or network."""

import numpy as np

from helpers.autocal import H, W, Z
from birdbrain import config, watch, zones


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
