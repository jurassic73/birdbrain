"""Shared helpers for the presence view tests (used by more than one test module)."""
import functools

import cv2
import numpy as np

from birdbrain import classifier, zones


# The live birdbath (2026-11): the outer rim spans x 0.302..0.634 of the width.
LIVE_RING = (0.4683, 0.5347, 0.1659, 0.1175, 0.1345, 0.0966, 0.5224, 0.4681)


LIVE_CROP = (0.248, 0.30, 0.4547, 0.4991)       # 873 x 539 px on 1920x1080; the old center cut kept x 0.335..0.615


NARROW_CROP = (0.335, 0.30, 0.28, 0.50)         # just that kept strip


GROUND_CROP = (0.0, 0.5, 1.0, 0.5)              # 1920 x 540 px: very wide


GROUND_POLY = ((0.1, 0.5), (0.9, 0.5), (0.9, 1.0), (0.1, 1.0))


BIG, SMALL = (1080, 1920, 3), (540, 960, 3)


RIM_LEFT = {"x": 0.31, "y": 0.5347, "r": 0.015}  # on the rim's left edge: inside the bowl square, in the old cut strip


def need(name):
    """A name the contract promises on classifier; a missing one is an AttributeError in the test that uses it."""
    return getattr(classifier, name)


def live(**kw) -> "zones.Zones":
    return zones.Zones(**{"ring": LIVE_RING, "crop": LIVE_CROP, "ground": GROUND_POLY, "ground_crop": GROUND_CROP, **kw})


@functools.lru_cache(maxsize=None)
def _noise(shape, seed) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def noise(shape, seed=0) -> np.ndarray:
    """Random pixels, drawn once per shape and seed and copied (a 1080p draw is ~20 ms, the copy ~3 ms)."""
    return _noise(tuple(shape), seed).copy()


def fill_color() -> np.ndarray:
    return np.rint(np.asarray(need("PAD_FILL"), float)).astype(np.uint8)[:3]


def jpeg(img) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def centered(patch, box_img) -> tuple[int, int]:
    """Where the box's pixels sit in a padded patch (within a pixel or two of the middle): (top, left)."""
    bh, bw = box_img.shape[:2]
    dh, dw = patch.shape[0] - bh, patch.shape[1] - bw
    inner = box_img[3:-3, 3:-3]
    for oy in range(max(dh // 2 - 2, 0), dh // 2 + 3):
        for ox in range(max(dw // 2 - 2, 0), dw // 2 + 3):
            if np.array_equal(patch[oy + 3:oy + bh - 3, ox + 3:ox + bw - 3], inner):
                return oy, ox
    raise AssertionError("the box's pixels are not unchanged in the middle of the patch")
