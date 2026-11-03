"""One visit per stay: a bird bathing for three minutes is one visit, not three.

After a real visit the watcher keeps the visit open while the visitor is still in view, and holds new triggers
until it leaves. "Still there" compares each substream frame with the empty scene from just before the visit
(the running background at the trigger), so a crow standing still still counts. Changes that don't move for
STAY_STILL seconds join the empty scene: a crow flips leaves over as it walks, and on real frames a patch of
shuffled leaves was as big as a crow on the far lawn.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config, zones

GONE = "gone"    # nothing visitor-sized left for STAY_GAP seconds
LIGHT = "light"  # most of the view changed (a cloud, the IR switching): can't tell, so close the visit
LONG = "long"    # STAY_MAX reached
WET_EDGE = 6     # substream px: a change this close to the open water still counts as on the water


class Stay:
    def __init__(self, visit_id: int, empty: np.ndarray, mask: np.ndarray, night: bool, now: float):
        self.visit_id, self.night, self.started = visit_id, night, now
        self.empty = empty.astype("uint8")  # blurred gray full frame of the scene without the visitor
        self.mask = mask                    # where a visitor can be: rim, water, rocks and ground
        self.area = max(cv2.countNonZero(mask), 1)
        self.min_px = config.STAY_MIN_BLOB * mask.size
        self.seen = now    # last frame with the visitor in view
        self.moved = now   # last frame where something visitor-sized moved: when it left, for the record

    def blobs(self, gray: np.ndarray) -> np.ndarray | None:
        """255 on every visitor-sized change from the empty scene; None if most of the view changed."""
        return changes(gray, self.empty, self.mask, self.min_px, self.area)

    def present(self, gray: np.ndarray) -> bool | None:
        b = self.blobs(gray)
        return None if b is None else bool(cv2.countNonZero(b))

    def update(self, gray: np.ndarray, motion: np.ndarray, night: bool, now: float) -> str | None:
        """Feed one frame (blurred gray, like MotionDetector's) and where it changed since the last few frames
        (255 = moving). Returns why the stay ended, else None."""
        if night != self.night:
            return LIGHT
        b = self.blobs(gray)
        if b is None:
            return LIGHT
        if cv2.countNonZero(b):
            near = cv2.dilate(b, np.ones((9, 9), np.uint8))
            if cv2.countNonZero(cv2.bitwise_and(motion, near)) >= self.min_px / 4:
                self.moved = now
            if now - self.moved < config.STAY_STILL:
                self.seen = now
            else:
                self.empty = gray.copy()  # it has all sat still for a minute: leaves, not a visitor
        if now - self.started >= config.STAY_MAX:
            return LONG
        if now - self.seen >= config.STAY_GAP:
            return GONE
        return None


def changes(gray: np.ndarray, empty: np.ndarray, mask: np.ndarray, min_px: float, area: int = 0) -> np.ndarray | None:
    """255 on every change of at least min_px pixels from the empty scene inside the mask; None if most of the
    masked area changed (the light, not a visitor)."""
    changed = cv2.threshold(cv2.absdiff(gray, empty), config.STAY_DIFF, 255, cv2.THRESH_BINARY)[1]
    changed = cv2.bitwise_and(changed, mask)
    if cv2.countNonZero(changed) / (area or max(cv2.countNonZero(mask), 1)) > config.STAY_LIGHT:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    return np.isin(labels, big).astype(np.uint8) * 255


def like_substream(jpeg: bytes, shape: tuple[int, int]) -> np.ndarray:
    """A main-stream frame made comparable with the empty scene: substream size, gray, blurred like
    MotionDetector's frames."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (21, 21), 0)


def box_mask(shape: tuple[int, int], box) -> np.ndarray:
    """255 inside an x,y,w,h box of frame fractions (the whole frame when None; nothing for a box outside it)."""
    m = np.zeros(shape, np.uint8)
    if px := zones.pixel_box(shape, box or (0, 0, 1, 1)):
        x0, y0, x1, y1 = px
        m[y0:y1, x0:x1] = 255
    return m


def count_animals(jpeg: bytes, empty: np.ndarray, mask: np.ndarray, box=None, water=None) -> int | None:
    """How many separate visitor-sized changes from the empty scene the photo shows inside the crop box, for
    the local species gate (one BioCLIP guess can only name one animal). Changes closer than GROUP_GAP are one
    animal (a bird and its reflection or splash). A change that lies wholly on open water (`water`) doesn't
    count: sky reflections and drifting leaves move there, and a bird in the water nearly always touches the rim
    or the rocks. None when it can't tell (the light changed)."""
    gray = like_substream(jpeg, empty.shape)
    area = mask & box_mask(empty.shape, box)
    b = changes(gray, empty, area, config.STAY_MIN_BLOB * empty.size)
    if b is None:
        return None
    grouped = b
    if config.GROUP_GAP:
        k = 2 * config.GROUP_GAP + 1
        grouped = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) & area
    n, labels = cv2.connectedComponents(grouped)
    if water is None:
        return n - 1
    # The blur spreads a change on the water a few px onto the rim, so the water is widened by WET_EDGE.
    wet = cv2.dilate(water, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WET_EDGE + 1, 2 * WET_EDGE + 1)))
    return len(np.unique(labels[(b > 0) & (wet == 0)]))
