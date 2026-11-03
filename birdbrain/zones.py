"""The calibrated zones in use: the .env values, unless automatic recalibration has saved newer ones.

Recalibration (birdbrain.autocal) stores its zones in the settings table along with the .env values they grew
from. Editing the zones in .env by hand (a new manual calibration) makes those stale, so .env wins again.
No OpenCV here: the web process imports this.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from . import config, db, sun

KEY = "zones"


@dataclass(frozen=True)
class Zones:
    ring: tuple | None = None         # rim and water ellipses, see config.RING_FORM
    spots: tuple = ()                 # rocks in the water: (cx,cy,rx,ry[,angle]), ...
    roi: tuple | None = None          # x,y,w,h; only used without a ring
    crop: tuple | None = None         # what Claude sees for birdbath visits
    ground: tuple = ()                # ground polygon corners: (x,y), ...
    ground_crop: tuple | None = None  # what Claude sees for ground visits

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Zones":
        tup = lambda v: tuple(v) if v else None  # noqa: E731
        return cls(
            ring=tup(d.get("ring")),
            spots=tuple(tuple(s) for s in d.get("spots") or ()),
            roi=tup(d.get("roi")),
            crop=tup(d.get("crop")),
            ground=tuple(tuple(p) for p in d.get("ground") or ()),
            ground_crop=tup(d.get("ground_crop")),
        )

    def shifted(self, dx: float, dy: float) -> "Zones":
        """Every zone moved by dx, dy (fractions of the frame), for when the camera itself has moved. Nothing
        leaves the frame: a box or polygon pushed over an edge keeps only the part still in view (dropped when
        none is) and an ellipse's centre stays on the frame. A coordinate below 0 would otherwise slice the frame
        from its far side and hand the watcher an empty image on every trigger."""
        def box(b):
            return clip_box((b[0] + dx, b[1] + dy, *b[2:])) if b else None

        def ellipse(e):
            return (unit(e[0] + dx), unit(e[1] + dy), *e[2:])

        ring = None
        if self.ring:
            outer, inner = config.ring_ellipses(self.ring)
            ring = config.make_ring(ellipse(outer), ellipse(inner))
        return Zones(
            ring=ring,
            spots=tuple(ellipse(s) for s in self.spots),
            roi=box(self.roi),
            crop=box(self.crop),
            ground=clip_polygon((x + dx, y + dy) for x, y in self.ground),
            ground_crop=box(self.ground_crop),
        )


def from_env() -> Zones:
    return Zones(ring=config.MOTION_RING, spots=config.MOTION_SPOTS, roi=config.MOTION_ROI, crop=config.CROP_ROI,
                 ground=config.GROUND_ZONE, ground_crop=config.GROUND_CROP)


def _env_snapshot() -> dict:
    return json.loads(json.dumps(from_env().to_dict()))  # tuples as lists, to compare with stored JSON


def stored(conn) -> dict | None:
    """The saved recalibration, or None when there is none or .env has been recalibrated by hand since."""
    raw = db.get_value(conn, KEY)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if data.get("env") == _env_snapshot() else None


def load(conn) -> Zones:
    data = stored(conn)
    return Zones.from_dict(data["zones"]) if data else from_env()


def save(conn, zones: Zones, manual: bool | None = None) -> None:
    """manual: the owner set these by hand, so they become the anchor later automatic runs measure against.
    None (automatic saves) keeps the last hand-set anchor; False drops it."""
    data = {"env": _env_snapshot(), "zones": zones.to_dict()}
    old = stored(conn)
    if manual:
        data["manual"] = zones.to_dict()
    elif manual is None and old and old.get("manual"):
        data["manual"] = old["manual"]
    db.set_value(conn, KEY, json.dumps(data))
    forget()


def manual(conn) -> Zones | None:
    """The zones the owner last set by hand (Adjust zones), or None."""
    data = stored(conn)
    return Zones.from_dict(data["manual"]) if data and data.get("manual") else None


def unit(v) -> float:
    """A frame fraction held to 0..1."""
    return round(min(max(float(v), 0.0), 1.0), 4)


def clamp_box(b):
    if not b:
        return None
    x, y, w, h = (min(max(float(v), 0.0), 1.0) for v in b)
    w, h = min(max(w, 0.02), 1 - x), min(max(h, 0.02), 1 - y)
    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))


def clip_box(b):
    """The part of an x,y,w,h box that lies inside the frame; None when none of it does."""
    if not b:
        return None
    x, y, w, h = (float(v) for v in b)
    x0, y0, x1, y1 = max(x, 0.0), max(y, 0.0), min(x + w, 1.0), min(y + h, 1.0)
    if x1 <= x0 or y1 <= y0:
        return None
    return clamp_box((x0, y0, x1 - x0, y1 - y0))


def clip_polygon(points) -> tuple:
    """A polygon's corners held inside the frame; () when what is left has no area (pushed off the edge)."""
    pts = tuple((unit(x), unit(y)) for x, y in points)
    if len(set(pts)) < 3:
        return ()
    twice_area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    return pts if abs(twice_area) / 2 >= 1e-4 else ()


def pixel_box(shape, box) -> tuple[int, int, int, int] | None:
    """An x,y,w,h box of frame fractions as pixel bounds (x0, y0, x1, y1) on a frame of `shape` (H, W, ...),
    held inside the frame; None when nothing of the box is in it. Slicing with the raw values would wrap a
    negative edge round to the far side of the frame and return nothing."""
    H, W = shape[:2]
    x, y, w, h = box
    x0, y0 = max(int(x * W), 0), max(int(y * H), 0)
    x1, y1 = min(int(round((x + w) * W)), W), min(int(round((y + h) * H)), H)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def edited(current: Zones, d: dict) -> Zones:
    """current with the shapes the owner dragged (ring, spots, crop, ground), clamped to the frame.
    A ground crop that was just the polygon's box follows the polygon."""
    f = lambda v, lo=0.0, hi=1.0: round(min(max(float(v), lo), hi), 4)  # noqa: E731

    def ellipse(e, lo, hi):  # center on the frame, radii in range, angle as -90..90 degrees
        cx, cy, rx, ry, angle = config.ellipse5(e)
        angle = (float(angle) + 90) % 180 - 90
        return (f(cx), f(cy), f(rx, lo, hi), f(ry, lo, hi), round(angle, 1))
    ring = current.ring
    if d.get("ring") and current.ring:
        r = d["ring"]
        if not 6 <= len(r) <= 10:
            raise ValueError("ring")
        outer, inner = config.ring_ellipses(r)
        ring = config.make_ring(ellipse(outer, 0.005, 0.5), ellipse(inner, 0.005, 0.5))
    spots = current.spots
    if "spots" in d:
        if any(len(s) not in (4, 5) for s in d["spots"]):
            raise ValueError("spots")
        spots = tuple(e if e[4] else e[:4] for e in (ellipse(s, 0.003, 0.3) for s in d["spots"]))
    ground, ground_crop = current.ground, current.ground_crop
    if d.get("ground") and current.ground:
        if len(d["ground"]) < 3:
            raise ValueError("ground")
        ground = tuple((f(x), f(y)) for x, y in d["ground"])
        if ground_crop and clamp_box(config.polygon_box(current.ground)) == clamp_box(ground_crop):
            ground_crop = clamp_box(config.polygon_box(ground))
    crop = clamp_box(d["crop"]) if d.get("crop") and current.crop else current.crop
    return Zones(ring=ring, spots=spots, roi=current.roi, crop=crop, ground=ground, ground_crop=ground_crop)


def objects(z: Zones) -> list[dict]:
    """The zones as shapes for the editor (the same form as scene objects): the bath's rim and its water as two
    ellipses, each rock spot, the ground polygon and the crop box as 4 corners. Ellipses carry their angle in
    degrees as a fifth number. `role` says which zone each one is."""
    out = []
    if z.ring:
        outer, inner = config.ring_ellipses(z.ring)
        out.append({"role": "ring", "name": "Bath rim", "kind": "birdbath", "ellipse": list(outer)})
        out.append({"role": "water", "name": "Open water", "kind": "pool", "ellipse": list(inner)})
        for i, s in enumerate(z.spots):
            name = "Rock spot" + (f" {i + 1}" if len(z.spots) > 1 else "")
            out.append({"role": "spot", "name": name, "kind": "landing_spot", "ellipse": list(config.ellipse5(s))})
    if z.ground:
        out.append({"role": "ground", "name": "Ground", "kind": "ground", "points": [list(p) for p in z.ground]})
    if z.crop:
        x, y, w, h = z.crop
        out.append({"role": "crop", "name": "What Claude sees", "kind": "crop",
                    "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]})
    return out


def from_objects(objs: list[dict]) -> dict:
    """The editor's shapes back into the form `edited` takes. The rim and the water make the ring (a rim with an
    `inner` is the older single-shape form); a crop box is the box around its corners; a spot drawn as points becomes the ellipse around them."""
    def box(pts):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def ellipse(o):
        if o.get("ellipse"):
            return list(o["ellipse"])
        x1, y1, x2, y2 = box(o["points"])
        return [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2]

    d: dict = {"spots": []}
    outer = inner = None
    for o in objs:
        role = o.get("role")
        if role == "ring":
            outer = ellipse(o)
            inner = o.get("inner") or inner
        elif role == "water":
            inner = ellipse(o)
        elif role == "spot":
            d["spots"].append(ellipse(o))
        elif role == "ground":
            d["ground"] = o["points"]
        elif role == "crop":
            x1, y1, x2, y2 = box(o["points"]) if o.get("points") else (
                o["ellipse"][0] - o["ellipse"][2], o["ellipse"][1] - o["ellipse"][3],
                o["ellipse"][0] + o["ellipse"][2], o["ellipse"][1] + o["ellipse"][3])
            d["crop"] = [x1, y1, x2 - x1, y2 - y1]
        else:
            raise ValueError("role")
    if outer:
        cx, cy, orx, ory, angle = config.ellipse5(outer)
        d["ring"] = list(config.make_ring(outer, inner or (cx, cy, orx * 0.8, ory * 0.8, angle)))
    return d


def reset(conn) -> None:
    """Back to the .env values."""
    conn.execute("DELETE FROM settings WHERE key = ?", (KEY,))
    conn.commit()
    forget()


def reference_path(night: bool = False) -> Path:
    """The median frame new ones are lined up against to notice the camera moving, one per light (IR edges
    differ). Named after the .env zones, so a hand calibration starts a fresh reference."""
    tag = hashlib.sha1(json.dumps(_env_snapshot(), sort_keys=True).encode()).hexdigest()[:8]
    return config.DATA_DIR / "calibration" / f"reference-{tag}{'-night' if night else ''}.png"


def clear_reference() -> None:
    for night in (False, True):
        reference_path(night).unlink(missing_ok=True)


# --- When to recalibrate (Settings page) ---

DEFAULT_TIMES = ("10:30", "14:30")  # daylight all year, after the morning glare off the water
RUN_WINDOW = timedelta(hours=2)     # a slot missed by more than this (watcher down) waits for the next one


SUN_OFFSET = timedelta(hours=1)  # "sunrise" runs this long after sunrise, "sunset" this long before sunset
SUN_SLOTS = ("sunrise", "sunset")


def slot_time(entry: str, day) -> datetime | None:
    """When a schedule entry ("14:30", "sunrise" or "sunset") falls on day; None if the sun times are unknown."""
    if entry == "sunrise":
        t = sun.sunrise(day)
        return t + SUN_OFFSET if t else None
    if entry == "sunset":
        t = sun.sunset(day)
        return t - SUN_OFFSET if t else None
    hour, minute = map(int, entry.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute)


def describe(entry: str, day) -> str:
    """"14:30", or "1 h after sunrise (08:39)" with that day's time."""
    if entry not in SUN_SLOTS:
        return entry
    when = slot_time(entry, day)
    text = f"1 h {'after' if entry == 'sunrise' else 'before'} {entry}"
    return f"{text} ({when:%H:%M})" if when else f"{text} (location not set)"


def schedule(conn) -> dict:
    raw = db.get_value(conn, "zones_times")
    times = [t for t in (raw.split(",") if raw is not None else DEFAULT_TIMES) if t]
    today = datetime.now().date()
    return {"auto": db.get_value(conn, "zones_auto") != "0", "times": times,
            "labels": [describe(t, today) for t in times], "located": sun.located(),
            "sun": {e: f"{w:%H:%M}" for e in SUN_SLOTS if (w := slot_time(e, today))}}


def save_schedule(conn, auto: bool, times: list[str]) -> None:
    clean = []
    for t in times:
        t = t.strip()
        if t in SUN_SLOTS:
            clean.append(t)
        elif t:
            datetime.strptime(t, "%H:%M")  # ValueError for anything else
            clean.append(t)
    db.set_value(conn, "zones_auto", "1" if auto else "0")
    order = lambda t: (t == "sunset", t != "sunrise", t)  # noqa: E731  sunrise first, sunset last
    db.set_value(conn, "zones_times", ",".join(sorted(set(clean), key=order)))


def request(conn) -> None:
    """Ask the watcher to recalibrate as soon as it can (it checks every CHECK_EVERY seconds)."""
    db.set_value(conn, "zones_request", datetime.now().isoformat(timespec="seconds"))


def pending(conn) -> bool:
    """A requested recalibration the watcher hasn't finished yet."""
    req = db.get_value(conn, "zones_request")
    last = db.recent_calibrations(conn, 1)
    return bool(req) and (not last or last[0]["ran_at"] < req)


DURATIONS_KEPT = 10


def note_finished(conn, reason: str, ran_at: str) -> None:
    """Remember how long an asked-for run took (from the button press), for the Zones panel's timer."""
    req = db.get_value(conn, "zones_request")
    if reason != "manual" or not req or ran_at < req:
        return
    secs = (datetime.fromisoformat(ran_at) - datetime.fromisoformat(req)).total_seconds()
    db.set_value(conn, "zones_durations", json.dumps((durations(conn) + [round(secs)])[-DURATIONS_KEPT:]))


def durations(conn) -> list[int]:
    try:
        return [int(s) for s in json.loads(db.get_value(conn, "zones_durations") or "[]")]
    except (ValueError, TypeError):
        return []


def status(conn) -> dict:
    """What the Zones panel's timer needs: is a run pending, since when, and how long runs usually take."""
    past = durations(conn)
    return {"pending": pending(conn), "started": db.get_value(conn, "zones_request"),
            "average": round(sum(past) / len(past)) if past else None}


def due(conn, now: datetime) -> str | None:
    """"manual" if recalibration was asked for, "scheduled" if a daily slot has come up, else None."""
    if pending(conn):
        return "manual"
    s = schedule(conn)
    if not s["auto"]:
        return None
    last = db.recent_calibrations(conn, 1)
    last_at = datetime.fromisoformat(last[0]["ran_at"]) if last else None
    for t in s["times"]:
        slot = slot_time(t, now.date())
        if slot and slot <= now < slot + RUN_WINDOW and (last_at is None or last_at < slot):
            return "scheduled"
    return None


_cache: tuple[float, Zones | None] = (0.0, None)


def current() -> Zones:
    """The zones in use, re-read at most every 30 s (the watcher asks on every frame)."""
    global _cache
    read_at, zones = _cache
    if zones is None or time.monotonic() - read_at > 30:
        with db.session() as conn:
            zones = load(conn)
        _cache = (time.monotonic(), zones)
    return zones


def forget() -> None:
    global _cache
    _cache = (0.0, None)
