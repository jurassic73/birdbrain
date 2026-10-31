"""Automatic zone recalibration: keeps the rock spots on the rocks, and every zone on the bath if the camera moves.

Runs inside the watcher, which already decodes the substream (the camera only serves a few streams), at the
times set on the Settings page or right away when asked there. It samples quiet frames for about a minute
and takes their per-pixel median, so ripples, drifting leaves and visiting birds drop out. Then:

1. Camera shift: the bath's edges are lined up against a reference median (phase correlation), one for day and
   one for night, since IR lights the bath differently. A clear shift of a few pixels moves every zone with it.
   A big one is only reported: that needs a hand calibration.
2. Rocks: by day the water reflects the sky and the rocks are dark; at night the water is black and the IR
   lights the rocks up. Each rock spot moves onto the matching cluster near it, keeping about its size.
   Shapes far from every spot (a sunken leaf) are ignored.
"""

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime

import cv2
import numpy as np

from . import config, db, zones as zone_store
from .zones import Zones

log = logging.getLogger("birdbrain.autocal")

CHECK_EVERY = 20       # seconds between checks for a due or requested recalibration
SAMPLE_EVERY = 2.0     # seconds between sampled frames
SAMPLES = 30           # frames in the median (~1 minute when the bath is quiet)
MIN_SAMPLES = 10       # fewer than this by MAX_WAIT: too busy, try at the next slot
MAX_WAIT = 600         # seconds

WATER_MARGIN = (0.88, 0.85)  # shrink the water ellipse away from the bowl's dark inner wall
ROCK_DARK = 35               # how much darker (by day) or brighter (at night) than the water a rock pixel must be
ROCK_MIN_AREA = 0.0001       # smallest dark cluster, as a fraction of the frame (~23 px on the substream)
SPOT_SEARCH = 2.5            # look for a spot's rocks within this many spot radii
SPOT_PAD = 1.3               # the spot is the rocks' box plus this margin, for a bird standing on them
SPOT_SIZE_RANGE = (0.6, 1.5) # never shrink or grow a spot past this share of its size before
MOVE_MIN_PX = 3              # smaller moves are noise; the zones stay exactly as they were
SHIFT_MIN_RESPONSE = 0.3     # phase correlation peak; the same view scores ~0.65
SHIFT_MIN_RESPONSE_CROSS = 0.08  # night frame against a day reference: a moved IR view scored 0.10 to 0.15
CROSS_FIT_GAIN = 1.2         # ...so that shift must also put the bath outline on 20% stronger edges
SHIFT_MAX = 0.08             # a camera move bigger than this share of the frame width needs a hand calibration
KEEP_IMAGES = 30


@dataclass
class Outcome:
    status: str     # moved, unchanged or skipped
    summary: str
    zones: Zones
    camera_moved: bool = False


def median_frame(frames: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def water_area(shape: tuple[int, int], ring) -> np.ndarray:
    H, W = shape
    cx, cy, _, _, irx, iry = ring[:6]
    icy = ring[6] if len(ring) > 6 else cy
    mask = np.zeros((H, W), np.uint8)
    axes = (int(irx * W * WATER_MARGIN[0]), int(iry * H * WATER_MARGIN[1]))
    cv2.ellipse(mask, (int(cx * W), int(icy * H)), axes, 0, 0, 360, 255, -1)
    return mask


def dark_clusters(gray: np.ndarray, ring, night: bool = False) -> list[dict]:
    """Rock-like clusters in the water, biggest first, in pixels: x, y, w, h, area, cx, cy. Darker than the
    water by day, brighter at night (IR lights the rocks, the water stays black)."""
    H, W = gray.shape
    water = water_area(gray.shape, ring)
    if not cv2.countNonZero(water):
        return []
    # The water's own brightness, from a wide median: sky reflections vary across the bowl, and unlike a
    # morphological close, a median isn't pulled up by a bright floating leaf.
    fill = gray.copy()
    fill[water == 0] = int(np.median(gray[water > 0]))
    k = max(int(0.1 * W) | 1, 3)
    bg = cv2.medianBlur(fill, k)
    dark = cv2.subtract(gray, bg) if night else cv2.subtract(bg, gray)
    m = cv2.bitwise_and(cv2.threshold(dark, ROCK_DARK, 255, cv2.THRESH_BINARY)[1], water)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # Glints split one pile of rocks into pieces; join pieces a few pixels apart before counting.
    joined = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, labels, _, _ = cv2.connectedComponentsWithStats(joined)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero((labels == i) & (m > 0))
        if len(xs) < ROCK_MIN_AREA * H * W:
            continue
        x, y, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        out.append({"x": x, "y": y, "w": x2 - x, "h": y2 - y, "area": len(xs), "cx": (x + x2) / 2, "cy": (y + y2) / 2})
    return sorted(out, key=lambda c: c["area"], reverse=True)


def direction(dx: float, dy: float) -> str:
    parts = []
    if abs(dx) >= 1:
        parts.append("right" if dx > 0 else "left")
    if abs(dy) >= 1:
        parts.append("down" if dy > 0 else "up")
    return " and ".join(parts) or "in place"


def track_spots(spots, clusters: list[dict], shape: tuple[int, int], anchor=None) -> tuple[tuple, list[str]]:
    """Each rock spot moved onto the biggest dark cluster near it, with notes on what happened.
    anchor: the spots as the owner last set them by hand. Those are trusted: a spot keeps its hand-set size
    and only moves when the rocks have left it (their middle is outside the spot)."""
    H, W = shape
    if anchor is not None and len(anchor) != len(spots):
        anchor = None
    new, notes, used = [], [], set()
    for n, spot in enumerate(spots, 1):
        name = "The rock spot" if len(spots) == 1 else f"Rock spot {n}"
        sx, sy, srx, sry = (spot[0] * W, spot[1] * H, spot[2] * W, spot[3] * H)
        reach = SPOT_SEARCH * max(srx, sry)
        near = [i for i, c in enumerate(clusters) if i not in used and math.hypot(c["cx"] - sx, c["cy"] - sy) <= reach]
        if not near:
            new.append(tuple(spot))
            notes.append(f"{name}: no rocks seen near it (under water, or covered?), so it was left alone.")
            continue
        i = max(near, key=lambda i: clusters[i]["area"])
        used.add(i)
        c = clusters[i]
        if anchor is not None:
            hrx, hry = anchor[n - 1][2] * W, anchor[n - 1][3] * H
            dx, dy = c["cx"] - sx, c["cy"] - sy
            if (dx / srx) ** 2 + (dy / sry) ** 2 <= 1 and abs(hrx - srx) < 2 and abs(hry - sry) < 2:
                new.append(tuple(spot))
                notes.append(f"{name}: still on the rocks, where you put it.")
                continue
            new.append((round(c["cx"] / W, 4), round(c["cy"] / H, 4), round(hrx / W, 4), round(hry / H, 4)))
            notes.append(f"{name}: the rocks moved {math.hypot(dx, dy):.0f} px {direction(dx, dy)}; "
                         "followed them, keeping the size you set.")
            continue
        lo, hi = SPOT_SIZE_RANGE
        rx = min(max(c["w"] / 2 * SPOT_PAD + 3, lo * srx), hi * srx)
        ry = min(max(c["h"] / 2 * SPOT_PAD + 3, lo * sry), hi * sry)
        dx, dy = c["cx"] - sx, c["cy"] - sy
        moved = math.hypot(dx, dy)
        if moved < MOVE_MIN_PX and abs(rx - srx) < 2 and abs(ry - sry) < 2:
            new.append(tuple(spot))
            notes.append(f"{name}: still on the rocks.")
            continue
        new.append((round(c["cx"] / W, 4), round(c["cy"] / H, 4), round(rx / W, 4), round(ry / H, 4)))
        notes.append(f"{name}: moved {moved:.0f} px {direction(dx, dy)} onto the rocks." if moved >= MOVE_MIN_PX
                     else f"{name}: resized to fit the rocks.")
    if ignored := len(clusters) - len(used):
        notes.append(f"Ignored {ignored} other dark shape{'s' if ignored != 1 else ''} in the water (leaves?).")
    return tuple(new), notes


def edge_image(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def bath_box(shape: tuple[int, int], ring) -> tuple[int, int, int, int]:
    """Pixel bounds around the bath, where the edges are strong and steady; the whole frame without a ring."""
    H, W = shape
    if not ring:
        return 0, 0, W, H
    cx, cy, orx, ory = ring[:4]
    x0, x1 = max(int((cx - 1.6 * orx) * W), 0), min(int((cx + 1.6 * orx) * W), W)
    y0, y1 = max(int((cy - 1.8 * ory) * H), 0), min(int((cy + 1.8 * ory) * H), H)
    return x0, y0, x1, y1


def camera_shift(reference: np.ndarray, gray: np.ndarray, ring) -> tuple[float, float, float]:
    """How far the view moved since the reference (dx, dy in pixels) and how sure (the correlation peak)."""
    x0, y0, x1, y1 = bath_box(gray.shape, ring)
    a, b = edge_image(reference)[y0:y1, x0:x1], edge_image(gray)[y0:y1, x0:x1]
    (dx, dy), response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F))
    return dx, dy, response


def ring_fit(gray: np.ndarray, ring) -> float:
    """Mean edge strength along the bath's outer and inner ellipses: higher when they sit on the real bath."""
    if not ring:
        return 0.0
    H, W = gray.shape
    cx, cy, orx, ory, irx, iry = ring[:6]
    icy = ring[6] if len(ring) > 6 else cy
    mask = np.zeros((H, W), np.uint8)
    for x, y, rx, ry in ((cx, cy, orx, ory), (cx, icy, irx, iry)):
        cv2.ellipse(mask, (int(x * W), int(y * H)), (int(rx * W), int(ry * H)), 0, 0, 360, 255, 2)
    return float(edge_image(gray)[mask > 0].mean())


def recalibrate(median: np.ndarray, zones: Zones, reference: np.ndarray | None, night: bool = False,
                cross: bool = False, anchor: Zones | None = None) -> Outcome:
    """New zones from a median frame (night: an IR one). cross: the reference is from the other light (no
    reference for this one yet), so the shift needs less correlation but must also fit the bath better. Pure."""
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    notes, z, camera_moved = [], zones, False
    light = "night" if night else "daytime"
    if reference is None or reference.shape != gray.shape:
        notes.append(f"Saved a {light} reference view of the bath, to notice if the camera moves later.")
    else:
        dx, dy, response = camera_shift(reference, gray, z.ring)
        dist = math.hypot(dx, dy)
        if cross:
            fits = response >= SHIFT_MIN_RESPONSE_CROSS and dist >= MOVE_MIN_PX and \
                ring_fit(gray, z.shifted(dx / W, dy / H).ring) >= CROSS_FIT_GAIN * ring_fit(gray, z.ring)
            response = SHIFT_MIN_RESPONSE if fits else 0.0
            if not fits and dist < MOVE_MIN_PX:
                response = SHIFT_MIN_RESPONSE  # lined up and not moved
            notes.append(f"Saved a {light} reference view of the bath.")
        if response >= SHIFT_MIN_RESPONSE and dist >= MOVE_MIN_PX:
            if dist > SHIFT_MAX * W:
                return Outcome("skipped", f"The camera seems to have moved a lot ({dist:.0f} px {direction(dx, dy)}). "
                                          "Zones left alone; recalibrate by hand with birdbrain.calibrate.", zones)
            z, camera_moved = z.shifted(dx / W, dy / H), True
            notes.append(f"The camera moved {dist:.0f} px {direction(dx, dy)}; every zone moved with it.")
        elif response >= SHIFT_MIN_RESPONSE:
            notes.append("The camera hasn't moved.")
        else:
            notes.append("Couldn't line the view up with the reference (bad light?), so the camera check was skipped.")
    if z.ring and z.spots:
        spots, spot_notes = track_spots(z.spots, dark_clusters(gray, z.ring, night), (H, W),
                                       anchor.spots if anchor else None)
        z = replace(z, spots=spots)
        notes += spot_notes
    return Outcome("moved" if z != zones else "unchanged", " ".join(notes), z, camera_moved)


# --- Pictures for the Settings page ---

def _draw(img: np.ndarray, z: Zones, color: tuple[int, int, int], thick: int) -> None:
    H, W = img.shape[:2]

    def ellipse(cx, cy, rx, ry):
        cv2.ellipse(img, (int(cx * W), int(cy * H)), (int(rx * W), int(ry * H)), 0, 0, 360, color, thick, cv2.LINE_AA)

    if z.ring:
        cx, cy, orx, ory, irx, iry = z.ring[:6]
        ellipse(cx, cy, orx, ory)
        ellipse(cx, z.ring[6] if len(z.ring) > 6 else cy, irx, iry)
    for s in z.spots:
        ellipse(*s)
    if z.ground:
        pts = np.array([(int(x * W), int(y * H)) for x, y in z.ground], np.int32)
        cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)


def picture(median: np.ndarray, before: Zones, after: Zones, scale: int = 3) -> np.ndarray:
    """A close-up of the bath from the median frame, the old zones in red and the new ones in green."""
    img = cv2.resize(median, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if after != before:
        _draw(img, before, (60, 60, 230), 1)
    _draw(img, after, (80, 220, 80), 2)
    if not after.ring:
        return img
    boxes = [bath_box(median.shape[:2], z.ring) for z in (before, after) if z.ring]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return img[y0 * scale:y1 * scale, x0 * scale:x1 * scale]


def picture_dir():
    return config.DATA_DIR / "calibration"


def _prune_pictures() -> None:
    pics = sorted(picture_dir().glob("cal_*.jpg"))
    for p in pics[:-KEEP_IMAGES]:
        p.unlink(missing_ok=True)


# --- Running inside the watcher ---

def run(frames: list[np.ndarray], reason: str, night: bool = False) -> None:
    """Recalibrate from sampled frames, save the result and keep a record of it."""
    median = median_frame(frames)
    with db.session() as conn:
        before = zone_store.load(conn)
        anchor = zone_store.manual(conn)
    ref_path = zone_store.reference_path(night)
    cross = not ref_path.exists()  # no reference in this light yet: line up against the other one
    src = zone_store.reference_path(not night) if cross else ref_path
    reference = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE) if src.exists() else None
    out = recalibrate(median, before, reference, night, cross and reference is not None, anchor)
    picture_dir().mkdir(parents=True, exist_ok=True)
    if out.camera_moved:  # the other light's reference shows the old view; it would shift the zones again
        zone_store.reference_path(not night).unlink(missing_ok=True)
    if out.status != "skipped" and (cross or out.camera_moved):
        cv2.imwrite(str(ref_path), cv2.cvtColor(median, cv2.COLOR_BGR2GRAY))  # new baseline (first run or moved)
    name = f"cal_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    cv2.imwrite(str(picture_dir() / name), picture(median, before, out.zones), [cv2.IMWRITE_JPEG_QUALITY, 85])
    _prune_pictures()
    with db.session() as conn:
        if out.status == "moved":
            zone_store.save(conn, out.zones)
        db.add_calibration(conn, ran_at=datetime.now().isoformat(timespec="seconds"), reason=reason,
                           status=out.status, summary=out.summary, before_json=_json(before),
                           after_json=_json(out.zones), image_path=name)
    log.info("recalibration (%s): %s. %s", reason, out.status, out.summary)


def _json(z: Zones) -> str:
    return json.dumps(z.to_dict())


def record_skip(reason: str, summary: str) -> None:
    with db.session() as conn:
        db.add_calibration(conn, ran_at=datetime.now().isoformat(timespec="seconds"), reason=reason,
                           status="skipped", summary=summary)
    log.info("recalibration (%s) skipped: %s", reason, summary)


class Recalibrator:
    """Fed every substream frame by the watcher; samples quiet daytime frames when a recalibration is due."""

    def __init__(self):
        self.reason: str | None = None  # set while sampling
        self.night: bool | None = None   # the light the sampling started in
        self.frames: list[np.ndarray] = []
        self.started = self.last_sample = self.next_check = 0.0
        self.working = threading.Event()

    def feed(self, frame: np.ndarray, night: bool, quiet: bool) -> None:
        now = time.time()
        if self.reason is None:
            if now < self.next_check or self.working.is_set():
                return
            self.next_check = now + CHECK_EVERY
            try:
                with db.session() as conn:
                    self.reason = zone_store.due(conn, datetime.now())
            except Exception:
                log.exception("couldn't check whether recalibration is due")
            if self.reason:
                self.frames, self.started, self.last_sample, self.night = [], now, 0.0, None
                log.info("recalibration (%s): sampling quiet frames", self.reason)
            return
        if self.night is None:
            self.night = night
        elif night != self.night:  # the IR switched mid-sample (dusk or dawn): the median would mix both
            record_skip(self._stop(), "The light changed while looking (the night vision switched). Trying again at the next time.")
            return
        if quiet and now - self.last_sample >= SAMPLE_EVERY:
            self.frames.append(frame.copy())
            self.last_sample = now
        if len(self.frames) >= SAMPLES or now - self.started > MAX_WAIT:
            frames, reason = self.frames, self._stop()
            if len(frames) < MIN_SAMPLES:
                record_skip(reason, "Too much going on at the bath to get a clear look. Trying again at the next time.")
                return
            self.working.set()
            threading.Thread(target=self._run, args=(frames, reason, self.night), daemon=True).start()

    def _stop(self) -> str:
        reason, self.reason, self.frames = self.reason, None, []
        return reason

    def _run(self, frames: list[np.ndarray], reason: str, night: bool | None) -> None:
        try:
            run(frames, reason, bool(night))
        except Exception:
            log.exception("recalibration failed")
            record_skip(reason, "Something went wrong; see the watcher log.")
        finally:
            self.working.clear()
