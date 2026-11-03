"""Circle from the motion: where the animal is in a photo the Pi named on its own.

Claude draws a circle around what it identified; the local species classifier (BioCLIP 2) only says what, not
where. So for a visit the Pi settles itself, the photo is compared with an empty view of the scene and the biggest
solid patch of change near the zone that fired becomes the circle, in the same form as Claude's (x, y as fractions
of the frame, r of its width) plus "pi": True. Plain OpenCV on a substream-sized gray copy: ~20 ms on a Mac.

The empty view:
- live: the motion detector's running background, snapshotted at the trigger (seconds old, same light);
- backfill (or without one): the per-pixel median of the nearest stored photos of the same view and light, which
  drops an animal that is in only one of them; else the zone calibration's reference view.

Brightness is matched between the two (exposure), and a change where the local pattern is the same, only lit
differently (a sun patch, a cloud), doesn't count. Open water is left out first (by day it mirrors the moving sky);
only when nothing is found elsewhere does one compact, solid patch on the water count (a bathing bird). Nothing
is stored when there is no clear answer: no patch, too big (the light), or two of similar size (which one?).

python -m birdbrain.motion_circle --backfill [--dry-run] [--limit N] [--redo]
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import cv2
import numpy as np

from . import config, db, zones as zone_store

log = logging.getLogger("birdbrain.motion_circle")

WORK = (640, 360)  # substream size: the empty scene's own size, and plenty to find an animal
BLUR = 21          # the motion detector's blur, so the photo matches its empty scene
PAD = 0.25         # the search reaches this far outside the crop box (of its size): a tail or crest may stick out
REACH = 10         # px: the patch must touch the zone that fired, or come this close (a bird's body sits above the rim)
SAME = 0.75        # local correlation above this (both patches textured): the same thing, lit differently
TEXTURE = 25.0     # local variance below this is too flat to judge the pattern
LIGHT = 0.4        # more of the search area changed than this: the light changed, not an animal
WATER_SOLID = 0.7  # a patch on the open water must be this solid (area / its hull) ...
WATER_MAX_R = 0.05  # ... and no bigger than this (of the width), and the only one
NEIGHBORS = 5      # stored photos in the median ...
NEIGHBOR_HOURS = 3  # ... from within this many hours


def gray(img: np.ndarray) -> np.ndarray:
    """Any frame (color or gray, any size) as the substream-sized, blurred gray image this module compares."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[::-1] != WORK:
        img = cv2.resize(img, WORK, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (BLUR, BLUR), 0)


def decode(jpeg: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)  # half size: 4x less to decode
    return None if img is None else gray(img)


def masks(z: zone_store.Zones, trigger: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(the zone whose motion fired, the open water), substream-sized; None where there's no such zone."""
    from .watch import ground_mask, motion_mask, water_mask  # the watcher imports this module
    shape = WORK[::-1]
    water = water_mask(shape, z.ring, z.spots) if z.ring else None
    if trigger == "ground" and z.ground:
        zone = ground_mask(shape, z.ground, z.ring)
    elif trigger == "water" and z.ring:
        zone = water
    elif z.ring:
        zone = motion_mask(shape, z.ring, z.spots)
    else:
        zone = None
    return zone, water


def _box(box, pad: float = 0.0) -> tuple[int, int, int, int]:
    W, H = WORK
    x, y, w, h = box or (0, 0, 1, 1)
    x0, y0 = max(x - pad * w, 0), max(y - pad * h, 0)
    x1, y1 = min(x + w + pad * w, 1), min(y + h + pad * h, 1)
    return int(x0 * W), int(y0 * H), int(round(x1 * W)), int(round(y1 * H))


def _blur(img: np.ndarray, k: int = 15) -> np.ndarray:
    return cv2.blur(img, (k, k))


def lit_differently(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """True where both images show the same textured pattern (high local correlation): only the light changed."""
    mf, mr = _blur(f), _blur(r)
    vf, vr = _blur(f * f) - mf * mf, _blur(r * r) - mr * mr
    ncc = (_blur(f * r) - mf * mr) / np.sqrt(np.maximum(vf, 1e-3) * np.maximum(vr, 1e-3))
    return (ncc > SAME) & (vf > TEXTURE) & (vr > TEXTURE)


def _patches(mask: np.ndarray, allowed: np.ndarray) -> list[dict]:
    """Groups of change (pieces closer than GROUP_GAP are one animal) that are big enough and touch `allowed`."""
    W, H = WORK
    k = 2 * config.GROUP_GAP + 1
    grouped = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels = cv2.connectedComponents(grouped)
    out = []
    for i in range(1, n):
        piece = (labels == i) & (mask > 0)
        area = int(piece.sum())
        if area < config.CIRCLE_MIN_BLOB * W * H or not (piece & allowed).any():
            continue
        pts = cv2.findNonZero(piece.astype(np.uint8))
        (cx, cy), rad = cv2.minEnclosingCircle(pts)
        hull = cv2.contourArea(cv2.convexHull(pts)) or 1.0
        out.append({"area": area, "cx": cx, "cy": cy, "r": rad, "solid": min(area / hull, 1.0)})
    return sorted(out, key=lambda p: p["area"], reverse=True)


def find(frame: np.ndarray, empty: np.ndarray, box, zone: np.ndarray | None = None,
         water: np.ndarray | None = None, bathing: bool = True) -> tuple[dict | None, dict]:
    """The circle around the biggest solid patch of change between `frame` and `empty` (both from gray()) near
    the crop box `box` and touching `zone`, and what was seen (for the record). (None, info) when unsure.
    bathing: with nothing found off the water, one compact patch on the open water counts (not for the ground)."""
    W, H = WORK
    X0, Y0, X1, Y1 = _box(box, PAD)
    bx0, by0, bx1, by1 = _box(box)
    f = frame[Y0:Y1, X0:X1].astype(np.float32)
    r = empty[Y0:Y1, X0:X1].astype(np.float32)
    wet = (water[Y0:Y1, X0:X1] > 0) if water is not None else np.zeros(f.shape, bool)
    land = ~wet if (~wet).any() else np.ones(f.shape, bool)
    # Exposure: bring the empty view to the photo's level and contrast (the camera adjusts between them).
    fq, rq = np.percentile(f[land], (25, 50, 75)), np.percentile(r[land], (25, 50, 75))
    gain = float(np.clip((fq[2] - fq[0]) / max(rq[2] - rq[0], 1.0), 0.5, 2.0))
    r = (r - rq[1]) * gain + fq[1]
    d = np.abs(f - r)
    threshold = max(float(config.CIRCLE_DIFF), 4 * float(np.median(d[land])))
    changed = (d >= threshold) & ~lit_differently(f, r)
    m = changed.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   # speckle
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # knit a body
    info: dict = {"threshold": round(threshold, 1), "changed": round(float((m > 0).mean()), 3)}
    if info["changed"] > LIGHT:
        return None, {**info, "why": "light"}
    in_box = np.zeros(f.shape, bool)
    in_box[by0 - Y0:by1 - Y0, bx0 - X0:bx1 - X0] = True  # must reach into the crop box
    allowed = in_box.copy()
    if zone is not None:
        near = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * REACH + 1, 2 * REACH + 1)))
        allowed &= near[Y0:Y1, X0:X1] > 0
    patches = _patches(np.where(wet, 0, m).astype(np.uint8), allowed)
    too_big = [p for p in patches if p["r"] / W > config.CIRCLE_MAX_R]
    patches = [p for p in patches if p["r"] / W <= config.CIRCLE_MAX_R]
    info["where"] = "land"
    if not patches and bathing and wet.any():  # a bird that hopped from the rim into the water
        on_water = [p for p in _patches(np.where(wet, m, 0).astype(np.uint8), in_box)
                    if p["solid"] >= WATER_SOLID and p["r"] / W <= WATER_MAX_R]
        if len(on_water) == 1:
            patches, info["where"] = on_water, "water"
    if not patches:
        return None, {**info, "why": "too big" if too_big else "nothing"}
    best = patches[0]
    if len(patches) > 1 and patches[1]["area"] >= config.CIRCLE_AMBIGUOUS * best["area"]:
        return None, {**info, "why": "two alike", "patches": len(patches)}
    info.update(area=round(best["area"] / (W * H), 4), patches=len(patches))
    circle = {"x": round((best["cx"] + X0) / W, 4), "y": round((best["cy"] + Y0) / H, 4),
              "r": round(max(best["r"] / W, 0.01), 4), "pi": True}
    return circle, info


def circle_for(jpeg: bytes, empty: np.ndarray | None, z: zone_store.Zones, trigger: str | None,
               source: str | None) -> tuple[dict | None, dict]:
    """Fail-soft: the circle for a photo and an empty view (source "empty": the motion detector's background,
    already blurred; else a plain gray frame of any size), and the record of the attempt for raw_json."""
    started = time.perf_counter()
    if empty is None:
        return None, {"found": False, "why": "no empty view", "source": None}
    try:
        frame = decode(jpeg)
        if frame is None:
            return None, {"found": False, "why": "unreadable photo", "source": source}
        if source == "empty":
            ref = cv2.resize(empty, WORK, interpolation=cv2.INTER_AREA) if empty.shape[::-1] != WORK else empty
        else:
            ref = gray(empty)
        zone, water = masks(z, trigger)
        crop = z.ground_crop if trigger == "ground" else z.crop
        circle, info = find(frame, ref, crop, zone, water, bathing=trigger != "ground")
    except Exception:
        log.exception("circle from the motion failed")
        return None, {"found": False, "why": "error", "source": source}
    return circle, {**info, "source": source, "found": circle is not None,
                    "seconds": round(time.perf_counter() - started, 3)}


def live(jpeg: bytes, empty: np.ndarray | None, z: zone_store.Zones, trigger: str | None, captured_at: str,
         night: bool) -> tuple[dict | None, dict]:
    """For the watcher: the circle for a visit it just named on the Pi, from the empty scene at the trigger (or,
    without one, from stored photos). Never raises."""
    if empty is not None:
        return circle_for(jpeg, empty, z, trigger, "empty")
    try:
        with db.session() as conn:
            img, source = best_empty(conn, -1, captured_at, night, z)
    except Exception:
        log.exception("no empty view for the circle")
        img, source = None, None
    return circle_for(jpeg, img, z, trigger, source)


def _zones_of(conn, raw: dict, captured_at: str) -> zone_store.Zones | None:
    for d in (raw.get("zones"), db.zones_at(conn, captured_at)):
        if isinstance(d, dict):
            try:
                return zone_store.Zones.from_dict(d)
            except (TypeError, ValueError):
                pass
    return None


def _same_view(a: zone_store.Zones | None, b: zone_store.Zones | None) -> bool:
    """Whether two photos were taken with the camera in the same place, going by where the bath was."""
    if not a or not b or not a.ring or not b.ring:
        return a == b
    return abs(a.ring[0] - b.ring[0]) < 0.01 and abs(a.ring[1] - b.ring[1]) < 0.01


def stored_empty(conn, visit_id: int, captured_at: str, night: bool, z: zone_store.Zones | None) -> np.ndarray | None:
    """The median of the nearest stored photos (any visit, NEIGHBOR_HOURS either side) of the same view and light:
    an animal in one of them is outvoted. None with fewer than 3."""
    rows = conn.execute(
        "SELECT id, captured_at, image_path, raw_json FROM visits WHERE id != ? AND is_night = ? "
        f"AND {db.FULL_FRAME} AND abs(julianday(captured_at) - julianday(?)) * 24 <= ? "
        "ORDER BY abs(julianday(captured_at) - julianday(?)) LIMIT 30",
        (visit_id, int(night), captured_at, NEIGHBOR_HOURS, captured_at)).fetchall()
    frames = []
    for row in rows:
        try:
            raw = json.loads(row["raw_json"] or "{}") or {}
        except ValueError:
            raw = {}
        if not _same_view(z, _zones_of(conn, raw if isinstance(raw, dict) else {}, row["captured_at"])):
            continue
        path = config.IMAGES_DIR / row["image_path"]
        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_2) if path.exists() else None
        if img is not None:
            frames.append(cv2.resize(img, WORK, interpolation=cv2.INTER_AREA))
        if len(frames) >= NEIGHBORS:
            break
    return np.median(np.stack(frames), axis=0).astype(np.uint8) if len(frames) >= 3 else None


def reference_empty(conn, night: bool, z: zone_store.Zones | None) -> np.ndarray | None:
    """The zone calibration's median view of the empty bath, if the camera is still where the photo was taken."""
    path = zone_store.reference_path(night)
    if not path.exists() or not _same_view(z, zone_store.load(conn)):
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def best_empty(conn, visit_id: int, captured_at: str, night: bool,
               z: zone_store.Zones | None) -> tuple[np.ndarray | None, str | None]:
    if (img := stored_empty(conn, visit_id, captured_at, night, z)) is not None:
        return img, "stored"
    if (img := reference_empty(conn, night, z)) is not None:
        return img, "reference"
    return None, None


def save(conn, visit_id: int, circle: dict | None, info: dict) -> None:
    """Store the attempt (and the circle, when there is one) with the visit. Labels are never touched."""
    raw = "json_set(COALESCE(raw_json, '{}'), '$.calls.motion_circle', json(?))"
    if circle:
        conn.execute(f"UPDATE visits SET raw_json = json_set({raw}, '$.calls.circle', json(?)) WHERE id = ?",
                     (json.dumps(info), json.dumps(circle), visit_id))
    else:  # a retry that finds nothing drops the Pi's earlier circle (never Claude's: see pending)
        conn.execute(f"UPDATE visits SET raw_json = json_remove({raw}, '$.calls.circle') WHERE id = ? AND "
                     "COALESCE(json_extract(raw_json, '$.calls.circle.claude'), 0) = 0", (json.dumps(info), visit_id))
    conn.commit()


def pending(conn, redo: bool = False) -> list:
    """Pi-settled visits with a photo and no circle that haven't been tried; with redo, also those tried before
    (their Pi circle is replaced). Never a visit with Claude's circle."""
    sql = ("SELECT * FROM visits WHERE json_extract(raw_json, '$.calls.settled_by') = 'classifier' "
           f"AND {db.FULL_FRAME} AND (json_extract(raw_json, '$.calls.circle') IS NULL "
           "OR json_extract(raw_json, '$.calls.circle.pi') = 1)")
    if not redo:
        sql += " AND json_extract(raw_json, '$.calls.motion_circle') IS NULL"
    return conn.execute(sql + " ORDER BY id").fetchall()


def backfill(limit: int | None = None, dry_run: bool = False, redo: bool = False) -> None:
    with db.session() as conn:
        db.init_db(conn)
        rows = pending(conn, redo)[:limit] if limit else pending(conn, redo)
        total, found, missed, started = len(rows), 0, 0, time.time()
        print(f"{total} Pi-named visit{'s' if total != 1 else ''} without a circle{' (dry run)' if dry_run else ''}")
        for i, v in enumerate(rows, 1):
            try:
                raw = json.loads(v["raw_json"] or "{}") or {}
            except ValueError:
                raw = {}
            z = _zones_of(conn, raw, v["captured_at"])
            path = config.IMAGES_DIR / v["image_path"]
            if z is None or v["photo_deleted_at"] or not path.exists():
                circle, info = None, {"why": "no photo or zones", "source": None, "found": False}
            else:
                empty, source = best_empty(conn, v["id"], v["captured_at"], bool(v["is_night"]), z)
                circle, info = circle_for(path.read_bytes(), empty, z, raw.get("trigger"), source)
            found, missed = found + bool(circle), missed + (not circle)
            print(f"  visit {v['id']} {v['captured_at']}: " + (
                f"circle at {circle['x']:.3f},{circle['y']:.3f} r {circle['r']:.3f} (from {info['source']})"
                if circle else f"no circle ({info.get('why')})"))
            if not dry_run:
                save(conn, v["id"], circle, info)
            if i % 10 == 0 or i == total:
                left = (time.time() - started) / i * (total - i)
                print(f"{i}/{total} done, {missed} no blob, ~{left:.0f} s left", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backfill", action="store_true", help="circle the Pi-named visits that have none")
    parser.add_argument("--dry-run", action="store_true", help="say what would be stored, store nothing")
    parser.add_argument("--limit", type=int, help="at most this many visits")
    parser.add_argument("--redo", action="store_true", help="also retry visits tried before")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.backfill:
        parser.print_help()
        return
    backfill(args.limit, args.dry_run, args.redo)


if __name__ == "__main__":
    main()
