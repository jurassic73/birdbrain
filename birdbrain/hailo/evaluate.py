"""Offline scorecard: run the Hailo detector over the stored visit photos and compare with Birdbrain's final labels.

Reads the production database read-only. Labels honour the owner's corrections (corrected_visitor over visitor);
a visit is real when its final visitor is not "none" (db.REAL, restated here without importing the watcher's
modules). Two views of each photo: "frame", the whole photo shrunk to the substream's width (what the live
detector sees, so the representative one), and "crop", the stored crop box at full resolution (what a detector on
Claude's crop would see). python -m birdbrain.hailo_detect --evaluate
"""

import json
import logging
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .. import config, zones
from . import geometry, store

log = logging.getLogger("birdbrain.hailo")

MODES = ("frame", "subcrop", "crop")
SUB_WIDTH = 640  # the substream's width: the live detector sees 640x360 frames
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
VISITORS = ("bird", "squirrel", "raccoon", "possum", "other")


def final_label(row) -> str:
    return row["corrected_visitor"] or row["visitor"]


def is_real(row) -> bool:
    return final_label(row) != "none" or (row["corrected_visitor"] is None and row["corrected_species"] is not None)


def visit_crop(row, fallback: zones.Zones):
    """The crop box in force when the visit was captured (stored in raw_json since the zones moved into the
    database), else the current one."""
    try:
        z = json.loads(row["raw_json"] or "{}").get("zones") or {}
        if z.get("crop"):
            return tuple(z["crop"])
    except ValueError:
        pass
    return fallback.crop


def views(img: np.ndarray, crop) -> dict[str, np.ndarray]:
    """The pictures scored per visit. frame: the photo at the substream's width (1080p and 960x540 photos both
    land at 640x360). subcrop: the crop box cut from that small frame (the same pixels the live detector gets with
    HAILO_VIEW=crop, upscaled by the engine's letterbox). crop: the crop box cut from the photo as stored."""
    h, w = img.shape[:2]
    out = {}
    if w > SUB_WIDTH:
        out["frame"] = cv2.resize(img, (SUB_WIDTH, int(round(h * SUB_WIDTH / w))), interpolation=cv2.INTER_AREA)
    else:
        out["frame"] = img
    if crop:
        px = zones.pixel_box(img.shape, crop)
        if px:
            x0, y0, x1, y1 = px
            out["crop"] = img[y0:y1, x0:x1]
        px = zones.pixel_box(out["frame"].shape, crop)
        if px:  # the crop box cut from the substream-sized frame: what the live detector sees with HAILO_VIEW=crop
            x0, y0, x1, y1 = px
            out["subcrop"] = out["frame"][y0:y1, x0:x1]
    return out


def best_animal(found: list[geometry.Detection]) -> geometry.Detection | None:
    return next((d for d in found if d.animal), None)


def summarize(found: list[geometry.Detection]) -> dict:
    best = best_animal(found)
    return {"best": [best.label, best.score] if best else None,
            "found": [[d.label, d.score] for d in found[:5]]}


def collect(engine, conn, images_dir: Path, fallback: zones.Zones, limit: int | None = None,
            progress=None) -> list[dict]:
    """One row per visit still having its photo: id, label, real, night and each view's detections."""
    rows = conn.execute(
        "SELECT id, captured_at, image_path, visitor, corrected_visitor, species, corrected_species, is_night, raw_json "
        "FROM visits WHERE photo_deleted_at IS NULL AND photo_level IS NOT 'thumb' ORDER BY id").fetchall()
    if limit:
        rows = rows[:limit]
    out, timings = [], Counter()
    for i, row in enumerate(rows, 1):
        path = images_dir / row["image_path"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        rec = {"id": row["id"], "at": row["captured_at"], "label": final_label(row), "real": is_real(row),
               "night": bool(row["is_night"]), "species": row["corrected_species"] or row["species"],
               "size": [img.shape[1], img.shape[0]], "modes": {}}
        for mode, view in views(img, visit_crop(row, fallback)).items():
            start = time.perf_counter()
            rec["modes"][mode] = summarize(engine.infer(view))
            timings[mode] += time.perf_counter() - start
        out.append(rec)
        if progress and (i % 50 == 0 or i == len(rows)):
            progress(i, len(rows))
    log.info("scored %d photos; seconds per view: %s", len(out),
             {m: round(t / max(len(out), 1), 3) for m, t in timings.items()})
    return out


# --- the scorecard (pure) ----------------------------------------------------------------------------------------


def _best_score(rec: dict, mode: str) -> float:
    best = rec["modes"].get(mode, {}).get("best")
    return best[1] if best else 0.0


def _best_label(rec: dict, mode: str) -> str | None:
    best = rec["modes"].get(mode, {}).get("best")
    return best[0] if best else None


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return {"n": len(values), "min": round(min(values), 2), "q1": round(q[0], 2), "median": round(q[1], 2),
            "q3": round(q[2], 2), "max": round(max(values), 2)}


def scorecard(rows: list[dict], mode: str, thresholds=THRESHOLDS) -> dict:
    """Recall per visitor type (day and night), false triggers ignored, precision, at each threshold on the best
    animal-like score; plus what COCO called each visitor at the lowest threshold and the score distributions."""
    real = [r for r in rows if r["real"] and mode in r["modes"]]
    false = [r for r in rows if not r["real"] and mode in r["modes"]]
    by_type = {v: [r for r in real if r["label"] == v] for v in VISITORS if any(r["label"] == v for r in real)}
    card = {"mode": mode, "real": len(real), "false": len(false),
            "false_day": sum(not r["night"] for r in false), "false_night": sum(r["night"] for r in false),
            "real_night": sum(r["night"] for r in real), "by_type": {v: len(rs) for v, rs in by_type.items()},
            "thresholds": [], "classes": {}, "scores": {}}
    for thr in thresholds:
        flagged = lambda r: _best_score(r, mode) >= thr  # noqa: E731
        tp = sum(flagged(r) for r in real)
        fp = sum(flagged(r) for r in false)
        row = {"threshold": thr, "recall": tp / len(real) if real else None,
               "flagged_real": tp, "flagged_false": fp,
               "ignored_false": len(false) - fp, "ignored_rate": (len(false) - fp) / len(false) if false else None,
               "ignored_false_day": sum(not flagged(r) for r in false if not r["night"]),
               "ignored_false_night": sum(not flagged(r) for r in false if r["night"]),
               "precision": tp / (tp + fp) if tp + fp else None,
               "recall_by_type": {v: sum(flagged(r) for r in rs) / len(rs) for v, rs in by_type.items()}}
        card["thresholds"].append(row)
    for v, rs in by_type.items():
        card["classes"][v] = dict(Counter(_best_label(r, mode) or "nothing" for r in rs).most_common())
    card["classes"]["false triggers"] = dict(Counter(_best_label(r, mode) or "nothing" for r in false).most_common())
    card["scores"]["real"] = _quantiles([_best_score(r, mode) for r in real])
    card["scores"]["false"] = _quantiles([_best_score(r, mode) for r in false])
    card["scores"]["false_flagged"] = _quantiles([s for r in false if (s := _best_score(r, mode)) > 0])
    return card


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def markdown(cards: dict[str, dict[str, dict]], meta: dict) -> str:
    """The scorecard report: one section per HEF, one table per view. cards: {model: {mode: scorecard}}."""
    lines = ["# Hailo-8 shadow detector: offline scorecard", "",
             f"Generated {meta.get('generated', '')} over {meta.get('photos', 0)} stored photos "
             f"({meta.get('real', 0)} real visits, {meta.get('false', 0)} false triggers that still have a photo).",
             "", "COCO has no squirrel, raccoon or possum class. \"Flagged\" means the best detection of any animal-like "
             "COCO class (" + ", ".join(sorted(geometry.ANIMAL_CLASSES)) + ") reached the threshold; the class itself "
             "is reported separately below. Real visits use the owner's corrected labels where there are any.", "",
             "Views: **frame** is the whole photo shrunk to the substream's width, the geometry the live detector "
             "sees with HAILO_VIEW=frame. **subcrop** is the crop box cut from that small frame and upscaled to the "
             "detector's input, what the live detector sees with HAILO_VIEW=crop (the default). Both are "
             "representative of the live service. **crop** is the stored crop box (what Claude sees) at full "
             "resolution, which shows what a detector on the main stream's crop could do.",
             ""]
    for model, modes in cards.items():
        lines += [f"## {model}", ""]
        if "seconds" in meta.get("timing", {}).get(model, {}):
            t = meta["timing"][model]
            lines += [f"Inference: {t['ms']:.1f} ms per view on the Hailo-8 (scoring took {t['seconds']:.0f} s).", ""]
        for mode, card in modes.items():
            lines += [f"### View: {mode}", "",
                      f"Real visits: {card['real']} ({', '.join(f'{v} {n}' for v, n in card['by_type'].items())}; "
                      f"{card['real_night']} at night). False triggers: {card['false']} "
                      f"({card['false_day']} day, {card['false_night']} night).", ""]
            types = list(card["by_type"])
            lines.append("| threshold | recall | " + " | ".join(types) + " | false ignored | day | night | precision |")
            lines.append("|---|---|" + "---|" * len(types) + "---|---|---|---|")
            for row in card["thresholds"]:
                lines.append(f"| {row['threshold']:.1f} | {_pct(row['recall'])} | "
                             + " | ".join(_pct(row["recall_by_type"].get(v)) for v in types)
                             + f" | {_pct(row['ignored_rate'])} | {row['ignored_false_day']}/{card['false_day']} | "
                               f"{row['ignored_false_night']}/{card['false_night']} | {_pct(row['precision'])} |")
            lines += ["", "What COCO called them (best animal-like class at the 0.2 floor, or nothing):", ""]
            for v, counts in card["classes"].items():
                lines.append(f"- {v}: " + ", ".join(f"{k} {n}" for k, n in counts.items()))
            lines += ["", "Best animal-like score per photo (0 = nothing found):", ""]
            for k, q in card["scores"].items():
                if q:
                    lines.append(f"- {k}: n {q['n']}, min {q['min']}, q1 {q['q1']}, median {q['median']}, "
                                 f"q3 {q['q3']}, max {q['max']}")
            lines.append("")
    return "\n".join(lines)


def run(hefs: list[Path], out: Path | None, limit: int | None = None) -> str:
    """Score every HEF over the stored photos, write the markdown report and return it."""
    from .engine import Engine

    fallback = zones.from_env()
    cards, timing, rows = {}, {}, []
    conn = store.production_db()
    try:
        try:
            fallback = zones.load(conn)
        except Exception:
            log.warning("could not read the zones from the database; using .env", exc_info=True)
        for hef in hefs:
            engine = Engine(hef, min_score=min(THRESHOLDS))
            try:
                start = time.perf_counter()
                rows = collect(engine, conn, config.IMAGES_DIR, fallback, limit,
                               progress=lambda i, n: print(f"  {engine.model}: {i}/{n} photos", flush=True))
                seconds = time.perf_counter() - start
                timing[engine.model] = {"seconds": seconds, "ms": engine.infer_ms}
                cards[engine.model] = {mode: scorecard(rows, mode) for mode in MODES}
                if out:  # the per-photo detections beside the report, for anyone who wants to dig
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.with_name(f"{out.stem}-{engine.model}.json").write_text(json.dumps(rows))
            finally:
                engine.close()
    finally:
        conn.close()
    meta = {"generated": datetime.now().replace(microsecond=0).isoformat(), "photos": len(rows),
            "real": sum(r["real"] for r in rows), "false": sum(not r["real"] for r in rows), "timing": timing}
    text = markdown(cards, meta)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return text
