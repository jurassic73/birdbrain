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

from . import config

GONE = "gone"    # nothing visitor-sized left for STAY_GAP seconds
LIGHT = "light"  # most of the view changed (a cloud, the IR switching): can't tell, so close the visit
LONG = "long"    # STAY_MAX reached


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
        changed = cv2.threshold(cv2.absdiff(gray, self.empty), config.STAY_DIFF, 255, cv2.THRESH_BINARY)[1]
        changed = cv2.bitwise_and(changed, self.mask)
        if cv2.countNonZero(changed) / self.area > config.STAY_LIGHT:
            return None
        n, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
        big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= self.min_px]
        return np.isin(labels, big).astype(np.uint8) * 255

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
