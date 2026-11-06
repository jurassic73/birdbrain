"""Shared helpers for the presence bowl tests (used by more than one test module)."""
import functools

import numpy as np


# The live zones on atmos (2026-11-04): the crop box is 873 x 539 px on a 1920 x 1080 frame.
RING = (0.4683, 0.5347, 0.1659, 0.1175, 0.1345, 0.0966, 0.5224, 0.4681)


CROP = (0.2477, 0.2236, 0.4543, 0.4982)


GROUND_CROP = (0.003, 0.0, 0.96, 0.9304)


LIVE = {"ring": list(RING), "spots": [], "roi": None, "crop": list(CROP), "ground": [], "ground_crop": list(GROUND_CROP)}


DIM = 16


@functools.lru_cache(maxsize=None)
def _noise(h, w, value):
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(value, 20, (h, w, 3)), 0, 255).astype(np.uint8)


def frame(h=1080, w=1920, value=60):
    """A noisy frame (seed 0). Drawn once per size and copied: a 1080p one takes ~0.15 s to draw, 3 ms to copy,
    and some tests paint on theirs."""
    return _noise(h, w, value).copy()


_noise(1080, 1920, 60)  # drawn once at import, so no single test pays the ~0.1 s for the first 1080p frame
