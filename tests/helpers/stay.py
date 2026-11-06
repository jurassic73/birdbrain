"""Shared helpers for the stay tests (used by more than one test module)."""
import cv2
import numpy as np

from birdbrain import stay


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
