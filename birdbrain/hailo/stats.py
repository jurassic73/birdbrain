"""The Stats page's "Hailo detector" card (admin only): health from the heartbeat, a scorecard for a window, the
detector-only events to review, and the offline scorecard's headline numbers. Read-only, cheap, fails to None.

No OpenCV or HailoRT here: the web process imports this. The join logic lives in birdbrain.hailo.report.
"""

import json
import logging
import re
import statistics
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from .. import config
from . import report, store

log = logging.getLogger("birdbrain.hailo")

WINDOWS = (("24h", "24 h"), ("3d", "3 days"), ("7d", "7 days"), ("all", "all"))
DEFAULT_WINDOW = "3d"
STALE_SECONDS = 60      # a heartbeat older than this: the service is not running (it writes every 5 s)
REVIEW_LIMIT = 20       # detector-only events listed, newest first
ALL_SINCE = "2000-01-01T00:00:00"
SNAPSHOT_TAGS = ("pre2s", "pre1s", "start", "best", "end")
SNAPSHOT_WORDS = {"pre2s": "2 s before", "pre1s": "1 s before", "start": "first frame", "best": "best frame",
                  "end": "after"}
# Exactly what store.snapshot_rel writes: a date folder, a time, the event id and a tag. Nothing else is served.
SNAPSHOT_RE = re.compile(r"\d{4}/\d{2}/\d{2}/\d{6}_\d+_[a-z0-9]+\.jpg")

# Headline numbers of docs/hailo-shadow-scorecard.md (2026-11-08, 487 stored photos: 210 birds, 71 squirrels, 206
# false triggers; yolov8s_h8), kept here because the docs folder is not deployed. Recall at the HEF's 0.2 floor and
# at 0.5, and the share of false triggers ignored at 0.2.
OFFLINE = {
    "photos": 487, "birds": 210, "squirrels": 71, "false": 206, "model": "yolov8s_h8",
    "views": (
        {"view": "whole substream frame", "recall": 23, "birds": 14, "squirrels": 52, "recall50": 16, "ignored": 99},
        {"view": "crop box from the substream (the live default)", "recall": 42, "birds": 39, "squirrels": 51,
         "recall50": 24, "ignored": 99},
        {"view": "crop box at full resolution", "recall": 48, "birds": 46, "squirrels": 54, "recall50": 31,
         "ignored": 95},
    ),
    "classes": "squirrels came out as bird 23, cat 9, dog 4, teddy bear 1, nothing 34 of 71",
}


def available() -> bool:
    """Whether there is anything to show: the detector's own database exists (never on a fresh install or the
    public mirror)."""
    try:
        return config.HAILO_SHADOW_DB.is_file()
    except OSError:
        return False


def window_since(key: str, now: datetime) -> str:
    """The ISO start of a window key; an unknown key is the default."""
    if key not in dict(WINDOWS):
        key = DEFAULT_WINDOW
    if key == "all":
        return ALL_SINCE
    return (now - report.parse_since(key)).replace(microsecond=0).isoformat()


def health(hb: dict | None, now: datetime) -> dict:
    """The health strip from the heartbeat: state is running, stale or missing."""
    if not hb or not hb.get("at"):
        return {"state": "missing", "age_s": None}
    try:
        age = (now - datetime.fromisoformat(hb["at"])).total_seconds()
    except ValueError:
        return {"state": "missing", "age_s": None}
    return {
        "state": "running" if age <= STALE_SECONDS else "stale",
        "age_s": int(max(age, 0)),
        "fps": hb.get("fps"), "infer_ms": hb.get("infer_ms"), "cpu_pct": hb.get("cpu_pct"),
        "chip_temp_c": hb.get("chip_temp_c"), "model": hb.get("model"), "view": hb.get("view") or config.HAILO_VIEW,
        "frames": hb.get("frames"), "uptime_min": (hb.get("uptime_s") or 0) // 60, "night": hb.get("night"),
        "errors": hb.get("errors"), "reconnects": hb.get("reconnects"),
        "other_classes": hb.get("other_classes") or {},
    }


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = min(int(round(p * (len(values) - 1))), len(values) - 1)
    return values[k]


def scorecard(result: dict) -> dict:
    """The window's numbers from report.correlate: agreement with real visits, lead times, suppressions, and how
    COCO's classes map onto Birdbrain's final labels (the four-component vocabulary calls that label Claude's or
    yours)."""
    matched, missed = result["matched"], result["missed"]
    leads = [m["lead_s"] for m in matched]
    ahead = [x for x in leads if x > 0]
    mapping: Counter = Counter()
    for m in matched:
        mapping[(m["visit"]["label"], m["event"]["label"])] += 1
    for a in result["agreed"]:
        mapping[("false trigger", a["event"]["label"])] += 1
    classes = [{"visit": v, "coco": c, "n": n} for (v, c), n in sorted(mapping.items(), key=lambda kv: -kv[1])]
    real = result["real"]
    false = len(result["suppressed"]) + len(result["agreed"])
    return {
        "events": result["events"], "animal_events": result["animal_events"],
        "in_zone": sum(1 for e in result["detector_only"]) + sum(1 for m in matched if m["event"]["in_zone"]),
        "visits": result["visits"], "real": real, "false": false,
        "matched": len(matched), "agreement": round(100 * len(matched) / real) if real else None,
        "missed": len(missed),
        "lead_median": statistics.median(leads) if leads else None, "lead_p90": _percentile(leads, 0.9),
        "ahead": len(ahead), "leads": len(leads),
        "suppressed": len(result["suppressed"]),
        "suppressed_pct": round(100 * len(result["suppressed"]) / false) if false else None,
        "agreed": len(result["agreed"]),
        "detector_only": len(result["detector_only"]),
        "classes": classes,
    }


def review_list(result: dict, limit: int = REVIEW_LIMIT) -> list[dict]:
    """Detector-only events, newest first, each with its arrival snapshots in order (only files the route
    will serve: valid names, and never the "after" shot)."""
    out = []
    for e in sorted(result["detector_only"], key=lambda e: e["started_at"], reverse=True)[:limit]:
        shots = []
        for rel in e.get("snapshots") or []:
            tag = rel.rsplit("_", 1)[-1].removesuffix(".jpg")
            if SNAPSHOT_RE.fullmatch(rel) and tag in SNAPSHOT_TAGS[:4]:
                shots.append({"rel": rel, "tag": tag, "word": SNAPSHOT_WORDS.get(tag, tag)})
        out.append({"id": e["id"], "started_at": e["started_at"], "ended_at": e["ended_at"], "label": e["label"],
                    "score": e["score"], "zone": e["zone"], "frames": e["frames"], "seconds": e.get("seconds"),
                    "night": e.get("night"), "shots": shots})
    return out


def load_events(conn, since: str) -> list[dict]:
    events = [dict(row) for row in store.events_since(conn, since)]
    for e in events:
        e["started_at"] = datetime.fromisoformat(e["started_at"])
        e["ended_at"] = datetime.fromisoformat(e["ended_at"]) if e["ended_at"] else None
        e["snapshots"] = json.loads(e["snapshots"]) if e["snapshots"] else []
        e["animal"], e["in_zone"] = bool(e["animal"]), bool(e["in_zone"])
    return events


def load_visits(conn, since: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, captured_at, ended_at, visitor, corrected_visitor, corrected_species FROM visits "
        "WHERE captured_at >= ? ORDER BY captured_at", (since,)).fetchall()
    visits = []
    for row in rows:
        label = row["corrected_visitor"] or row["visitor"]
        real = label != "none" or (row["corrected_visitor"] is None and row["corrected_species"] is not None)
        visits.append({"id": row["id"], "captured_at": datetime.fromisoformat(row["captured_at"]),
                       "ended_at": datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
                       "real": real, "label": label})
    return visits


def card(conn, window: str = DEFAULT_WINDOW, now: datetime | None = None) -> dict | None:
    """Everything the card renders, or None when there is nothing to show or the detector's database can't be
    read right now (the Stats page then simply leaves the card out). `conn` is the web app's (production) connection,
    used for the visits side of the join."""
    if not available():
        return None
    now = now or datetime.now()
    if window not in dict(WINDOWS):
        window = DEFAULT_WINDOW
    since = window_since(window, now)
    try:
        shadow = store.shadow_ro()
        try:
            events = load_events(shadow, since)
        finally:
            shadow.close()
        visits = load_visits(conn, since)
    except Exception:
        log.warning("the Hailo detector's database could not be read; leaving its card out", exc_info=True)
        return None
    result = report.correlate(events, visits)
    return {
        "window": window, "windows": WINDOWS, "since": since,
        "health": health(store.read_heartbeat(), now),
        "score": scorecard(result),
        "review": review_list(result), "review_limit": REVIEW_LIMIT,
        "offline": OFFLINE,
        "snapshot_dir": str(config.HAILO_SNAPSHOT_DIR),
    }


def snapshot_path(rel: str) -> Path | None:
    """The file behind /hailo/snapshot/<rel>, or None: only names of the exact shape the detector writes, resolved
    inside the snapshot folder (so no "..", no symlink out, nothing but those JPEGs)."""
    if not SNAPSHOT_RE.fullmatch(rel):
        return None
    root = config.HAILO_SNAPSHOT_DIR
    try:
        path = (root / rel).resolve(strict=True)
        if not path.is_relative_to(root.resolve()):
            return None
    except (OSError, RuntimeError):
        return None
    return path if path.is_file() else None


def thumbnail(path: Path, width: int = 320) -> bytes:
    """A small JPEG of a snapshot for the review list (never the full frame)."""
    import io

    from PIL import Image

    with Image.open(path) as img:
        img.draft("RGB", (width, width))  # cheap JPEG downscale on load
        img = img.convert("RGB")
        if img.width > width:
            img = img.resize((width, max(int(round(img.height * width / img.width)), 1)))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=78)
    return buf.getvalue()
