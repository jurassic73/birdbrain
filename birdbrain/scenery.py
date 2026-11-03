"""Scenery check: is there anything in this frame that isn't part of the usual empty scene?

The local detector also reports every object-like box with its objectness, whatever its class (a squirrel
scores as an object even though COCO has no squirrel). Rocks, the rim and leaf piles score too, but they
sit in the same place frame after frame. So each box is compared with the boxes in recent empty frames
from the same zone and time of day (day/night): a box that turned up in SCENERY_HITS of them is scenery.
The visit's `novel` score is the highest objectness left over; below NOVEL_MIN the frame looks empty.

Checked 2026-10-27 on the first 452 visits (each scored against the others): 99% of the empty frames
scored under 0.2, and all 10 real visitors scored 0.42 or more.
CLI: python -m birdbrain.scenery --backfill   (adds objects to visits stored before this existed)
"""

import argparse
import json

from . import config, db

NOVEL_MIN = 0.2       # novel objectness below this: nothing new in the frame
SCENERY_IOU = 0.4     # overlap with a box in an empty frame that counts as the same thing
SCENERY_HITS = 2      # empty frames a box must turn up in to be scenery
BACKGROUND = 300      # recent empty frames remembered per zone and day/night
MIN_BACKGROUND = 20   # fewer than this and there's no verdict yet


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = w * h
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def novel_score(objects, background) -> float:
    """Highest objectness among this frame's boxes that aren't scenery. `background` is a list of the
    object lists of empty frames; each object is [objectness, x, y, w, h]."""
    novel = 0.0
    for score, *box in objects:
        if score <= novel:
            continue
        hits = 0
        for frame in background:
            if any(iou(box, other[1:]) >= SCENERY_IOU for other in frame):
                hits += 1
                if hits >= SCENERY_HITS:
                    break
        if hits < SCENERY_HITS:
            novel = score
    return novel


def check(conn, det: dict | None, ground: bool, night: bool, exclude: int | None = None) -> dict | None:
    """{'novel', 'empty', 'background'} for a detector verdict, or None without objects or enough history."""
    if not det or "objects" not in det:
        return None
    background = db.empty_objects(conn, ground, night, BACKGROUND, exclude)
    if len(background) < MIN_BACKGROUND:
        return None
    novel = novel_score(det["objects"], background)
    return {"novel": round(novel, 3), "empty": novel < NOVEL_MIN, "background": len(background)}


def looks_empty(det: dict | None) -> bool:
    """True only when the scenery check ran and found nothing new."""
    return bool(det and (det.get("scenery") or {}).get("empty"))


def backfill() -> None:
    """Run the detector on stored visits that have no objects yet, then score every visit's scenery check
    against the others (so the Stats scorecard has history from day one)."""
    import cv2
    import numpy as np

    from . import detector
    from .watch import roi

    with db.session() as conn:
        rows = conn.execute("SELECT id, image_path, raw_json FROM visits ORDER BY id").fetchall()
    added = 0
    for r in rows:
        raw = json.loads(r["raw_json"] or "{}") or {}
        calls = raw.setdefault("calls", {})
        if "objects" in (calls.get("detector") or {}):
            continue
        img = cv2.imread(str(config.IMAGES_DIR / r["image_path"]))
        if img is None:
            continue
        img = cv2.resize(img, (1920, 1080))  # false triggers are stored smaller; crops are fractions
        crop = config.GROUND_CROP if raw.get("trigger") == "ground" else config.CROP_ROI
        verdict = detector.run(cv2.imencode(".jpg", img)[1].tobytes(), crop)
        if verdict is None:
            raise SystemExit("the local detector is off or its model is missing")
        det = calls.get("detector") or {}
        det["objects"] = verdict["objects"]
        calls["detector"] = det if det else verdict
        with db.session() as conn:
            db.set_raw_json(conn, r["id"], raw)
        added += 1
    print(f"objects added to {added} visits")
    scored = 0
    with db.session() as conn:
        for r in conn.execute("SELECT id, is_night, raw_json FROM visits").fetchall():
            raw = json.loads(r["raw_json"] or "{}") or {}
            det = (raw.get("calls") or {}).get("detector")
            if not det or "scenery" in det:
                continue
            result = check(conn, det, raw.get("trigger") == "ground", bool(r["is_night"]), exclude=r["id"])
            if result:
                det["scenery"] = {**result, "backfilled": True}
                db.set_raw_json(conn, r["id"], raw)
                scored += 1
    print(f"scenery check scored on {scored} visits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true", help="add objects and scenery scores to older visits")
    args = parser.parse_args()
    if not args.backfill:
        parser.error("nothing to do (try --backfill)")
    with db.session() as conn:
        db.init_db(conn)
    backfill()


if __name__ == "__main__":
    main()
