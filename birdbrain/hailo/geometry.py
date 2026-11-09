"""Pure pieces of the Hailo shadow detector: letterbox mapping, parsing the HEF's NMS output, tagging a box with
the calibrated zone it sits in, and the small IoU tracker that turns per-frame detections into events.

No HailoRT and no files here, so every function is unit-testable. Boxes are (x0, y0, x1, y1) as fractions of the
source frame unless a name says otherwise.
"""

import math
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import config
from ..detector import COCO

INPUT = 640  # the ready-made Hailo-8 YOLO HEFs take 640x640 RGB
PAD = 114    # YOLO's letterbox grey
# COCO has no squirrel, raccoon or possum: small mammals come out as cat, dog, bear or a bird. "Animal-like" is any of
# these; the raw class is kept beside it so the evaluation can say what each visitor was called.
ANIMAL_CLASSES = frozenset({"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                            "teddy_bear"})


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the source frame

    @property
    def animal(self) -> bool:
        return self.label in ANIMAL_CLASSES

    @property
    def center(self) -> tuple[float, float]:
        return box_center(self.box)


def letterbox_ratio(shape, size: int = INPUT) -> float:
    """The scale that fits a frame of `shape` (H, W, ...) inside size x size."""
    h, w = shape[:2]
    return min(size / h, size / w)


def letterbox(bgr: np.ndarray, size: int = INPUT) -> tuple[np.ndarray, float]:
    """A frame as the HEF's input: resized to fit, padded grey at the right and bottom, RGB, as a batch of one
    (1, size, size, 3) uint8. Returns it with the scale used, for parse_nms."""
    ratio = letterbox_ratio(bgr.shape, size)
    h, w = bgr.shape[:2]
    new_w, new_h = max(int(round(w * ratio)), 1), max(int(round(h * ratio)), 1)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR) if (new_w, new_h) != (w, h) else bgr
    padded = np.full((size, size, 3), PAD, np.uint8)
    padded[:new_h, :new_w] = resized
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)[None], ratio


def parse_nms(per_class, ratio: float, shape, min_score: float, size: int = INPUT,
              labels: tuple[str, ...] = tuple(COCO)) -> list[Detection]:
    """HailoRT's NMS-by-class output for one frame (a sequence of one (n, 5) array per class, rows y0, x0, y1, x1,
    score with the coordinates as fractions of the letterboxed input) as detections on the source frame of `shape`,
    best first. Boxes that fall in the grey padding are clipped to the frame."""
    h, w = shape[:2]
    found = []
    for ci, rows in enumerate(per_class):
        if ci >= len(labels):
            break
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, 5) if len(rows) else None
        if rows is None:
            continue
        for y0, x0, y1, x1, score in rows.tolist():
            if score < min_score:
                continue
            # letterbox fraction -> letterbox px -> source px -> source fraction
            fx0, fx1 = (min(max(v * size / ratio / w, 0.0), 1.0) for v in (x0, x1))
            fy0, fy1 = (min(max(v * size / ratio / h, 0.0), 1.0) for v in (y0, y1))
            if fx1 <= fx0 or fy1 <= fy0:
                continue
            found.append(Detection(labels[ci], round(float(score), 4), (round(fx0, 4), round(fy0, 4), round(fx1, 4),
                                                                        round(fy1, 4))))
    return sorted(found, key=lambda d: d.score, reverse=True)


# --- boxes and zones ---------------------------------------------------------------------------------------------


def box_center(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def box_area(b) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def boxes_overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def xywh_to_xyxy(box) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def to_frame(det: Detection, crop) -> Detection:
    """A detection made on the crop box's cut-out (its box as fractions of the cut-out) as fractions of the whole
    frame; crop is x,y,w,h of the frame, None for no crop."""
    if not crop:
        return det
    x, y, w, h = crop
    x0, y0, x1, y1 = det.box
    return Detection(det.label, det.score, (round(x + x0 * w, 4), round(y + y0 * h, 4), round(x + x1 * w, 4),
                                            round(y + y1 * h, 4)))


def point_in_ellipse(px: float, py: float, e) -> bool:
    """Whether a frame-fraction point lies inside an ellipse (cx, cy, rx, ry[, angle in degrees, clockwise])."""
    cx, cy, rx, ry, angle = config.ellipse5(e)
    if rx <= 0 or ry <= 0:
        return False
    a = math.radians(angle)
    dx, dy = px - cx, py - cy
    u = dx * math.cos(a) + dy * math.sin(a)
    v = -dx * math.sin(a) + dy * math.cos(a)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def box_touches_ellipse(box, e) -> bool:
    """A box overlaps an ellipse when its centre, a corner or an edge midpoint is inside it, or the ellipse's centre
    is inside the box. Good enough for tagging; nothing here needs an exact intersection."""
    x0, y0, x1, y1 = box
    cx, cy = config.ellipse5(e)[:2]
    if x0 <= cx <= x1 and y0 <= cy <= y1:
        return True
    mx, my = box_center(box)
    points = ((mx, my), (x0, y0), (x1, y0), (x0, y1), (x1, y1), (mx, y0), (mx, y1), (x0, my), (x1, my))
    return any(point_in_ellipse(px, py, e) for px, py in points)


def point_in_polygon(px: float, py: float, polygon) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        (ax, ay), (bx, by) = polygon[i], polygon[(i + 1) % n]
        if (ay > py) != (by > py):
            x = ax + (py - ay) * (bx - ax) / (by - ay)
            if px < x:
                inside = not inside
    return inside


BATH, CROP, GROUND, ELSEWHERE = "bath", "crop", "ground", "elsewhere"
IN_ZONE = (BATH, CROP)  # where the Motion detector's rim and perch area is; the two zones a visit is made from


def zone_of(box, zones) -> str:
    """Where a detection sits: "bath" when it touches the rim's outer ellipse (or the MOTION_ROI box without a
    ring), else "crop" inside the crop box Claude would see, else "ground" with its centre in the ground polygon,
    else "elsewhere". `zones` is a birdbrain.zones.Zones (ring, roi, crop, ground)."""
    if zones.ring:
        outer, _ = config.ring_ellipses(zones.ring)
        if box_touches_ellipse(box, outer):
            return BATH
    elif zones.roi and boxes_overlap(box, xywh_to_xyxy(zones.roi)):
        return BATH
    if zones.crop and boxes_overlap(box, xywh_to_xyxy(zones.crop)):
        return CROP
    if zones.ground and point_in_polygon(*box_center(box), zones.ground):
        return GROUND
    return ELSEWHERE


def frame_is_night(bgr: np.ndarray) -> bool:
    """IR night frames are grey, so colour saturation is near zero (the same rule as watch.frame_is_night, kept
    here so the detector's process never imports the watcher and its models)."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < 12


# --- tracking detections into events -----------------------------------------------------------------------------


@dataclass
class Track:
    """One object followed over frames. It becomes an event once it has persisted min_hits frames."""
    id: int
    box: tuple[float, float, float, float]
    first_t: float
    last_t: float
    hits: int = 1
    labels: Counter = field(default_factory=Counter)
    best: Detection | None = None
    best_t: float = 0.0
    zones: Counter = field(default_factory=Counter)
    confirmed: bool = False
    concurrent: int = 1  # the most confirmed tracks alive at once while this one lived (a crowd)

    def see(self, det: Detection, t: float, zone: str) -> None:
        self.box, self.last_t, self.hits = det.box, t, self.hits + 1
        self.labels[det.label] += 1
        self.zones[zone] += 1
        if self.best is None or det.score > self.best.score:
            self.best, self.best_t = det, t

    @property
    def label(self) -> str:
        return self.labels.most_common(1)[0][0] if self.labels else ""

    @property
    def zone(self) -> str:
        return self.zones.most_common(1)[0][0] if self.zones else ELSEWHERE

    @property
    def animal(self) -> bool:
        return any(label in ANIMAL_CLASSES for label in self.labels)

    @property
    def seconds(self) -> float:
        return max(self.last_t - self.first_t, 0.0)


class Tracker:
    """Greedy IoU matching, best score first. A detection that overlaps no track but whose centre lies within
    max_jump (a fraction of the frame) of a track's last box continues that track too: small birds move more than
    their own box between substream frames. A track unseen for gap seconds ends."""

    def __init__(self, min_iou: float = 0.3, max_jump: float = 0.06, min_hits: int = 3, gap: float = 2.0):
        self.min_iou, self.max_jump, self.min_hits, self.gap = min_iou, max_jump, min_hits, gap
        self.tracks: list[Track] = []
        self.next_id = 1

    def _match(self, det: Detection, free: list[Track]) -> Track | None:
        best, best_score = None, 0.0
        for tr in free:
            score = iou(det.box, tr.box)
            if score < self.min_iou:
                cx, cy = det.center
                tx, ty = box_center(tr.box)
                if math.hypot(cx - tx, cy - ty) > self.max_jump:
                    continue
                score = self.min_iou * 0.5  # a jump match, weaker than any real overlap
            if score > best_score:
                best, best_score = tr, score
        return best

    def update(self, detections: list[Detection], t: float, zones=None) -> tuple[list[Track], list[Track]]:
        """Feed one frame's detections at time t (seconds, monotonic). Returns the tracks that just became events
        and the events that just ended (tracks that never reached min_hits are dropped silently)."""
        free = list(self.tracks)
        for det in sorted(detections, key=lambda d: d.score, reverse=True):
            zone = zone_of(det.box, zones) if zones is not None else ELSEWHERE
            tr = self._match(det, free)
            if tr is None:
                tr = Track(self.next_id, det.box, t, t)
                tr.hits = 0
                self.next_id += 1
                self.tracks.append(tr)
            else:
                free.remove(tr)
            tr.see(det, t, zone)
        started = [tr for tr in self.tracks if not tr.confirmed and tr.hits >= self.min_hits]
        for tr in started:
            tr.confirmed = True
        alive = sum(1 for tr in self.tracks if tr.confirmed)
        for tr in self.tracks:
            if tr.confirmed:
                tr.concurrent = max(tr.concurrent, alive)
        ended = [tr for tr in self.tracks if t - tr.last_t > self.gap]
        for tr in ended:
            self.tracks.remove(tr)
        return started, [tr for tr in ended if tr.confirmed]

    def flush(self) -> list[Track]:
        """End every track (the stream was lost); returns the confirmed ones."""
        ended, self.tracks = [tr for tr in self.tracks if tr.confirmed], []
        return ended
