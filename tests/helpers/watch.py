"""Shared helpers for the watch tests (used by more than one test module)."""
import sqlite3

import cv2
import numpy as np

from birdbrain import watch


W, H = 640, 360


# A birdbath-like ring in the middle of the frame, with a rock spot inside the water.
RING = (0.5, 0.5, 0.2, 0.25, 0.15, 0.18)


SPOTS = ((0.42, 0.5, 0.03, 0.04),)


def frame(color=(90, 110, 80)) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = color
    return img


def with_blob(x: int, y: int, r: int = 14) -> np.ndarray:
    img = frame()
    cv2.circle(img, (x, y), r, (30, 30, 30), -1)
    return img


WATER_SPOT = (int(0.56 * W), int(0.53 * H))  # inside the water, away from the rim and the rock spot


def water_detector():
    d = watch.MotionDetector(None, 0.02, RING, SPOTS)
    for _ in range(5):
        d.score(frame())
    return d


def run_frames(d, frames) -> list[bool]:
    hits = []
    for f in frames:
        d.score(f)
        hits.append(d.water_hit)
    return hits


GROUND = ((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0))  # the bottom of the frame; the bath's lower edge dips in


GROUND_Y = int(0.88 * H)


def night_frame() -> np.ndarray:
    return frame((100, 100, 100))  # IR: no color


def walker(base, x: int, r: int, y: int = GROUND_Y) -> np.ndarray:
    img = base.copy()
    cv2.circle(img, (x, y), r, (20, 20, 20), -1)
    return img


def ground_detector(base) -> watch.MotionDetector:
    d = watch.MotionDetector(None, 0.02, RING, SPOTS, GROUND)
    for _ in range(5):
        d.score(base)
    return d


def ground_hits(d, frames) -> list[bool]:
    hits = []
    for f in frames:
        d.score(f)
        hits.append(d.ground_hit)
    return hits


def locked():
    return sqlite3.OperationalError("database is locked")
