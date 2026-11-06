"""Shared helpers for the autocal tests (used by more than one test module)."""
import os
import time

import cv2
import numpy as np
import pytest

from birdbrain import config, db, zones


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


def run_at(conn, when: str, status="unchanged"):
    db.add_calibration(conn, ran_at=when, reason="scheduled", status=status, summary="")


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
