"""Live correlation report: the shadow detector's events joined to the watcher's visits by time.

For the owner to read after a few days: which visits the detector saw first (and by how much), which events had no
visit at all (possible motion misses, with their snapshots to look at), which real visits it never saw, and which
false triggers it would have left alone. python -m birdbrain.hailo_detect --report --since 24h
"""

import json
import re
from datetime import datetime, timedelta

from .. import config
from . import store

BEFORE = 90   # seconds before captured_at an event may start and still belong to that visit
AFTER = 60    # seconds after a visit (or its stay's end) an event may still belong to it
MIN_FRAMES = 5  # a detector-only event shorter than this is not worth the owner's time


def parse_since(text: str) -> timedelta:
    """"24h", "3d", "90m" or "2w" as a timedelta."""
    m = re.fullmatch(r"\s*(\d+)\s*([mhdw])\s*", text.lower())
    if not m:
        raise ValueError(f"--since wants a number and m, h, d or w (e.g. 24h), not {text!r}")
    n, unit = int(m.group(1)), m.group(2)
    return timedelta(**{{"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[unit]: n})


def _dt(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


def correlate(events: list[dict], visits: list[dict], before: int = BEFORE, after: int = AFTER,
              min_frames: int = MIN_FRAMES) -> dict:
    """events: {id, started_at, ended_at (datetime or None), animal, in_zone, label, score, frames, snapshots}.
    visits: {id, captured_at, ended_at (datetime or None), real, label}. An animal event belongs to a visit when
    its span meets [captured_at - before, (ended_at or captured_at) + after]."""
    animal_events = [e for e in events if e["animal"]]
    used: set[int] = set()
    matched, missed, suppressed, agreed = [], [], [], []
    for v in sorted(visits, key=lambda v: v["captured_at"]):
        lo = v["captured_at"] - timedelta(seconds=before)
        hi = (v["ended_at"] or v["captured_at"]) + timedelta(seconds=after)
        hits = [e for e in animal_events if e["started_at"] <= hi and (e["ended_at"] or e["started_at"]) >= lo]
        used.update(e["id"] for e in hits)
        if v["real"]:
            if hits:
                first = min(hits, key=lambda e: e["started_at"])
                lead = (v["captured_at"] - first["started_at"]).total_seconds()
                matched.append({"visit": v, "event": first, "lead_s": round(lead, 1), "events": len(hits)})
            else:
                missed.append(v)
        elif hits:
            agreed.append({"visit": v, "event": max(hits, key=lambda e: e["score"] or 0)})
        else:
            suppressed.append(v)
    detector_only = [e for e in animal_events
                     if e["id"] not in used and e["in_zone"] and (e["frames"] or 0) >= min_frames]
    return {"matched": matched, "missed": missed, "suppressed": suppressed, "agreed": agreed,
            "detector_only": detector_only, "events": len(events), "animal_events": len(animal_events),
            "visits": len(visits), "real": sum(v["real"] for v in visits)}


def _when(d: datetime | None) -> str:
    return d.strftime("%m-%d %H:%M:%S") if d else "?"


def render(result: dict, since: str) -> str:
    r = result
    lines = [f"Hailo shadow detector vs the Motion detector, last {since}",
             f"  detector events: {r['events']} ({r['animal_events']} animal-like); visits: {r['visits']} "
             f"({r['real']} real, {r['visits'] - r['real']} false triggers)", ""]
    lines.append(f"Real visits the detector also saw: {len(r['matched'])} of {r['real']}")
    leads = [m["lead_s"] for m in r["matched"]]
    if leads:
        earlier = [x for x in leads if x > 0]
        lines.append(f"  detector ahead of the trigger on {len(earlier)}/{len(leads)}; lead median "
                     f"{sorted(leads)[len(leads) // 2]:.0f} s, max {max(leads):.0f} s (negative = detector later)")
    for m in r["matched"]:
        v, e = m["visit"], m["event"]
        lines.append(f"  visit {v['id']} {_when(v['captured_at'])} {v['label']:<9} event {e['id']} "
                     f"{e['label']} {e['score'] or 0:.2f} started {_when(e['started_at'])} lead {m['lead_s']:+.0f} s"
                     + (f" ({m['events']} events)" if m["events"] > 1 else ""))
    lines += ["", f"Real visits the detector never saw: {len(r['missed'])}"]
    for v in r["missed"]:
        lines.append(f"  visit {v['id']} {_when(v['captured_at'])} {v['label']}")
    lines += ["", f"Detector-only events in the bath or crop zone (possible motion misses; review the snapshots): "
                  f"{len(r['detector_only'])}"]
    for e in r["detector_only"]:
        snaps = e["snapshots"] or []
        lines.append(f"  event {e['id']} {_when(e['started_at'])} to {_when(e['ended_at'])} {e['label']} "
                     f"{e['score'] or 0:.2f}, {e['frames']} frames")
        for s in snaps:
            lines.append(f"      {config.HAILO_SNAPSHOT_DIR / s}")
    lines += ["", f"False triggers the detector would have left alone: {len(r['suppressed'])} of "
                  f"{len(r['suppressed']) + len(r['agreed'])}"]
    for a in r["agreed"]:
        v, e = a["visit"], a["event"]
        lines.append(f"  false trigger {v['id']} {_when(v['captured_at'])} also had event {e['id']} "
                     f"{e['label']} {e['score'] or 0:.2f}")
    return "\n".join(lines)


def load(since: timedelta, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    start = (now - since).replace(microsecond=0).isoformat()
    with store.session() as conn:
        events = [dict(row) for row in store.events_since(conn, start)]
    for e in events:
        e["started_at"], e["ended_at"] = _dt(e["started_at"]), _dt(e["ended_at"])
        e["snapshots"] = json.loads(e["snapshots"]) if e["snapshots"] else []
        e["animal"], e["in_zone"] = bool(e["animal"]), bool(e["in_zone"])
    conn = store.production_db()
    try:
        rows = conn.execute(
            "SELECT id, captured_at, ended_at, visitor, corrected_visitor, corrected_species FROM visits "
            "WHERE captured_at >= ? ORDER BY captured_at", (start,)).fetchall()
    finally:
        conn.close()
    visits = []
    for row in rows:
        label = row["corrected_visitor"] or row["visitor"]
        real = label != "none" or (row["corrected_visitor"] is None and row["corrected_species"] is not None)
        visits.append({"id": row["id"], "captured_at": _dt(row["captured_at"]), "ended_at": _dt(row["ended_at"]),
                       "real": real, "label": label})
    return correlate(events, visits)


def run(since_text: str) -> str:
    return render(load(parse_since(since_text)), since_text)
