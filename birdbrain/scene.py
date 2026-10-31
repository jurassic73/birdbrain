"""The scene: the objects birds use in the camera's view (a birdbath, its landing rocks, a feeder tray, a birdhouse).

The owner describes the scene in a few words on the Settings page. Claude (Sonnet) looks at a fresh main-stream
frame and outlines each object it finds, as fractions of the frame. That outline is a draft: the owner sees it
drawn on the frame and accepts it or tries again with a better description. Only accepted objects are kept.

Uses the owner's description from Claude's Notes (`scene_notes`). Stored in the settings table: `scene_draft` and
`scene` (JSON, see `Outline`); the accepted objects go into every identify prompt (`db.prompt_site`).
The SDK is imported lazily so the web app doesn't load it at startup.
"""

import json
import math
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from . import config, db, scene_fit

MODEL = "claude-sonnet-5-5"

# What each kind of object is, for the prompt and the page.
KINDS = {
    "birdbath": "a birdbath bowl: its rim, and the water inside",
    "landing_spot": "a place inside or on another object where birds stand, e.g. rocks in a birdbath",
    "feeder": "a feeder: a tray, platform, tube or suet cage",
    "birdhouse": "a nest box; outline its entrance hole",
    "perch": "a branch, post, rail or wire birds land on",
    "ground": "ground where birds and animals walk past or forage",
    "distraction": "something that moves or looks like an animal but isn't: a swaying branch, a wind chime, glare",
}
Kind = Literal["birdbath", "landing_spot", "feeder", "birdhouse", "perch", "ground", "distraction"]

PROMPT = """You set up a backyard wildlife camera. The owner describes what the camera looks at; you find each of
those objects in the photo and outline it, so the camera knows where to watch for birds and animals.

Kinds of object:
{kinds}

For each object give:
- name: a short name the owner would use, e.g. "birdbath", "rocks in the bath", "platform feeder".
- kind: one of the kinds above.
- shape: always "ellipse" for a birdbath, a landing spot and a birdhouse hole (round things look like ellipses
  at an angle), and for any other round thing; "polygon" for everything else.
- ellipse: center x, y and radii rx, ry as fractions of the photo's width and height (0 to 1, from the top left).
  For a birdbath, the outer edge of its rim. Null for polygons.
- inner: for a birdbath only, the edge of the water inside the rim, as an ellipse like the one above. Null otherwise.
- points: 3 to 12 corners as fractions of the width and height, in order around the outline. Empty for ellipses.
- note: one short line about it that helps tell animals from scenery later, e.g. "rocks poke out of the water
  when it is low and can look like a small dark bird".

Outline snugly: the outer edge of a bowl or tray, not the shadow around it. A birdbath is one object (its rim
and the water inside): don't list its rim or its water separately. Ground goes around a birdbath's base, never
across the bowl. List the objects the owner names,
plus anything else birds clearly use or anything that will set off a motion detector. Don't outline sky or
distant background. Then add notes: one or two lines on things that may cause false alarms (glare, reflections,
leaves, moving shadows) or anything in the description you couldn't find."""


class Point(BaseModel):
    x: float
    y: float


class Ellipse(BaseModel):
    cx: float
    cy: float
    rx: float
    ry: float


class SceneObject(BaseModel):
    name: str
    kind: Kind
    shape: Literal["ellipse", "polygon"]
    ellipse: Ellipse | None
    inner: Ellipse | None = None
    points: list[Point]
    note: str


class Outline(BaseModel):
    objects: list[SceneObject]
    notes: str


def clean(outline: Outline) -> list[dict]:
    """The objects as plain dicts, coordinates clamped to the frame; outlines with nothing to draw are dropped."""
    out = []
    for o in outline.objects:
        c = lambda v: round(min(max(v, 0.0), 1.0), 4)  # noqa: E731
        d = {"name": o.name.strip() or o.kind, "kind": o.kind, "note": o.note.strip()}
        if o.shape == "ellipse" and o.ellipse and o.ellipse.rx > 0 and o.ellipse.ry > 0:
            e = o.ellipse
            d["ellipse"] = [c(e.cx), c(e.cy), c(e.rx), c(e.ry)]
            if o.kind == "birdbath" and o.inner and o.inner.rx > 0 and o.inner.ry > 0:
                i = o.inner
                d["inner"] = [c(i.cx), c(i.cy), c(i.rx), c(i.ry)]
        elif len(o.points) >= 3:
            d["points"] = [[c(p.x), c(p.y)] for p in o.points]
        else:
            continue
        out.append(d)
    return out


def ask(jpeg: bytes, brief: str, location: str = "") -> tuple[Outline | None, dict]:
    """Claude's outline of the scene in one frame, plus token use."""
    from .identify import client, prepare

    text = f"The owner's description: {brief.strip() or '(none given; describe what you see)'}"
    if location:
        text += f"\nThe camera is in {location}."
    response = client().messages.parse(
        model=MODEL,
        max_tokens=4000,
        output_config={"effort": "medium"},
        system=PROMPT.format(kinds="\n".join(f"- {k}: {v}" for k, v in KINDS.items())),
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": prepare(jpeg)}},
            {"type": "text", "text": text},
        ]}],
        output_format=Outline,
    )
    usage = {"model": response.model, "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return None, {**usage, "stop_reason": response.stop_reason}
    return response.parsed_output, usage


def cost(usage: dict) -> float:
    price = db.USD_PER_MTOK["sonnet"]
    return (usage.get("input_tokens", 0) * price["input"] + usage.get("output_tokens", 0) * price["output"]) / 1e6


# --- Storage ---

def picture_dir():
    return config.DATA_DIR / "calibration"


def picture_name(name: str) -> bool:
    return bool(re.fullmatch(r"scene_\d{8}_\d{6}\.jpg", name))


def _load(conn, key: str) -> dict | None:
    raw = db.get_value(conn, key)
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def draft(conn) -> dict | None:
    return _load(conn, "scene_draft")


def accepted(conn) -> dict | None:
    return _load(conn, "scene")


def outline(conn, jpeg: bytes, text: str, location: str = "") -> dict:
    """Ask Claude for an outline, keep the frame it saw, and save the result as the draft (also on failure)."""
    now = datetime.now()
    name = f"scene_{now:%Y%m%d_%H%M%S}.jpg"
    picture_dir().mkdir(parents=True, exist_ok=True)
    (picture_dir() / name).write_bytes(jpeg)
    result, usage = ask(jpeg, text, location)
    raw = clean(result) if result else []
    objects = scene_fit.tidy(json.loads(json.dumps(raw)), jpeg) if result else []
    d = {"at": now.isoformat(timespec="seconds"), "brief": text.strip(), "image": name, "usage": usage,
         "usd": round(cost(usage), 4), "objects": objects, "claude": raw,  # Claude's shapes before fitting
         "notes": result.notes.strip() if result else "Claude didn't give an outline this time. Try again."}
    db.set_value(conn, "scene_draft", json.dumps(d))
    _prune(conn)
    return d


# --- The owner's adjustments (drag handles on the page) ---

SHAPE_KEYS = ("ellipse", "inner", "points", "pieces")
WHICH = {"draft": "scene_draft", "scene": "scene"}


def _frac(v) -> float:
    return round(min(max(float(v), 0.0), 1.0), 4)


def _check_shape(obj: dict, shape: dict) -> dict:
    """The new geometry for obj, clamped to the frame: an ellipse (with a bath ring's water ellipse) or points (with
    pieces). It may be the other form than obj has now (the owner switched it on the page); the water ellipse only goes
    with an outer one. Raises ValueError for anything else."""
    if ("ellipse" in shape) == ("points" in shape):
        raise ValueError("give an ellipse or points")
    out = {}
    for key in ("ellipse", "inner"):
        if key in shape:
            if len(shape[key]) != 4:
                raise ValueError(f"{key} needs x, y, rx, ry")
            e = [_frac(v) for v in shape[key]]
            if e[2] <= 0 or e[3] <= 0:
                raise ValueError("an ellipse needs a size")
            out[key] = e
    if "inner" in out and "ellipse" not in out:
        raise ValueError("water needs its rim")
    for key in ("points", "pieces"):
        if key in shape:
            polys = [shape[key]] if key == "points" else shape[key]
            if any(len(p) < 3 for p in polys):
                raise ValueError("an outline needs 3 corners or more")
            clean_polys = [[[_frac(x), _frac(y)] for x, y in p] for p in polys]
            out[key] = clean_polys[0] if key == "points" else clean_polys
    if "pieces" in out and "points" not in out:
        raise ValueError("pieces need points")
    return out


def set_shape(conn, which: str, index: int, shape: dict) -> dict:
    """Save the owner's adjusted geometry for one object; the first time, the automatic shape is kept for Reset."""
    d = _load(conn, WHICH[which])
    if not d or not 0 <= index < len(d["objects"]):
        raise LookupError("no such object")
    obj = d["objects"][index]
    new = _check_shape(obj, shape)
    obj.setdefault("auto", {k: obj[k] for k in SHAPE_KEYS if k in obj})
    for k in SHAPE_KEYS:  # the new geometry replaces the old, form and all
        obj.pop(k, None)
    obj.update(new)
    obj["edited"] = True
    db.set_value(conn, WHICH[which], json.dumps(d))
    return obj


CORNERS = 8  # an ellipse switched to points gets this many corners to drag (more crowd a small rock)


def convert_shape(conn, which: str, index: int, to: str) -> dict:
    """Switch one object between an ellipse and a polygon (the owner's choice). An ellipse becomes CORNERS points
    along it (a bath ring's water line is dropped); points become the ellipse around them."""
    if to not in ("ellipse", "points"):
        raise ValueError("unknown shape")
    d = _load(conn, WHICH[which])
    if not d or not 0 <= index < len(d["objects"]):
        raise LookupError("no such object")
    obj = d["objects"][index]
    if to in obj:
        return obj
    if to == "points":
        cx, cy, rx, ry = obj["ellipse"]
        t = [2 * math.pi * k / CORNERS for k in range(CORNERS)]
        new = {"points": [[_frac(cx + rx * math.cos(a)), _frac(cy + ry * math.sin(a))] for a in t]}
    else:
        pts = [p for ps in [obj["points"], *obj.get("pieces", [])] for p in ps]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        new = {"ellipse": [_frac((min(xs) + max(xs)) / 2), _frac((min(ys) + max(ys)) / 2),
                           max(_frac((max(xs) - min(xs)) / 2), 0.005), max(_frac((max(ys) - min(ys)) / 2), 0.005)]}
    obj.setdefault("auto", {k: obj[k] for k in SHAPE_KEYS if k in obj})
    for k in SHAPE_KEYS:
        obj.pop(k, None)
    obj.update(new)
    obj["edited"] = True
    db.set_value(conn, WHICH[which], json.dumps(d))
    return obj


def reset_shape(conn, which: str, index: int) -> None:
    """Put back the shape Claude and the fitting gave."""
    d = _load(conn, WHICH[which])
    if not d or not 0 <= index < len(d["objects"]):
        raise LookupError("no such object")
    obj = d["objects"][index]
    if "auto" in obj:
        auto = obj.pop("auto")
        for k in SHAPE_KEYS:  # it may have been switched to the other form since
            obj.pop(k, None)
        obj.update(auto)
    obj.pop("edited", None)
    db.set_value(conn, WHICH[which], json.dumps(d))


def accept(conn) -> bool:
    d = draft(conn)
    if not d or not d["objects"]:
        return False
    db.set_value(conn, "scene", json.dumps(d))
    discard(conn)
    return True


def discard(conn) -> None:
    conn.execute("DELETE FROM settings WHERE key = 'scene_draft'")
    conn.commit()
    _prune(conn)


def _prune(conn) -> None:
    """Keep only the frames the draft and the accepted scene were drawn on."""
    keep = {s["image"] for s in (draft(conn), accepted(conn)) if s}
    for p in picture_dir().glob("scene_*.jpg"):
        if p.name not in keep:
            p.unlink(missing_ok=True)
