"""What's in the owner's circle? The visit page lets the owner circle part of a photo (often a false trigger the
pipeline called empty) and ask, cheapest first:

1. the Pi: the local species classifier (BioCLIP 2, birdbrain.classifier) on just the circle's square, plus the
   local animal check's score (birdbrain.presence) on the same embedding. Free, ~6 to 22 s on a Pi 4.
2. Haiku, then Sonnet: the same square, with the identify prompt (Claude's Notes and the lessons), structured output.

Nothing here changes a label: each answer is offered on the page with a Use button, and saving still goes through
the correction form. Every check is kept in the visit's raw_json.calls.owner_checks, so the visit page can show it
and db.totals counts what the Claude ones cost.
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from . import classifier, config, db, identify, names, presence

SOURCES = ("pi", "haiku", "sonnet")
MODELS = {"haiku": identify.HAIKU, "sonnet": identify.SONNET}
MARGIN = 1.25  # the circle's bounding square plus a little room, so the whole animal is in it
SURE = 0.6  # the Pi's first guess must be at least this sure (among the local species) to say "Looks like"
KEEP = 20  # checks kept per visit, newest last
IDLE = 600  # s: the web process drops the 300 MB model after this long without a check
NOTE = ("This is a close crop of the part of the photo the owner circled; they think an animal may be in it. "
        "Answer about the animal in this crop.")


def _lower_priority() -> None:
    # On Linux nice() applies to the calling thread, and the model's own threads (made when it loads, in this
    # thread) inherit it, so a check never slows the ADS-B decoder on the Pi. Elsewhere it would renice the
    # whole web process, so it's skipped.
    if sys.platform.startswith("linux"):
        try:
            os.nice(10)
        except OSError:
            pass


# One check at a time: two at once would only fight over the Pi's cores.
POOL = ThreadPoolExecutor(1, thread_name_prefix="circle-check", initializer=_lower_priority)
_last_used = 0.0
_lock = threading.Lock()


def release_if_idle() -> None:
    """Free the model once nobody has asked for IDLE seconds (the watcher has its own copy)."""
    with _lock:
        if classifier._classifier is not None and time.monotonic() - _last_used >= IDLE:
            classifier._classifier = None


def owner_circle(text: str | None) -> tuple[int, dict] | None:
    """The circle to check, from the page's marks: the first one the owner drew (Claude's own are skipped),
    with its number on the photo."""
    return next(((i, m) for i, m in enumerate(db.parse_marks(text), 1) if not m.get("claude")), None)


def square(circle: dict, width: int, height: int, margin: float = MARGIN) -> tuple[float, float, float, float]:
    """The circle's bounding square times `margin`, as an x, y, w, h box of the frame (kept inside it)."""
    x0, y0, side = classifier.circle_square(circle, width, height, margin)
    return x0 / width, y0 / height, side / width, side / height


def visitor_of(name: str, scientific: str, mammals: set[str]) -> str:
    """The visit form's type for one of the classifier's names."""
    if scientific not in mammals:
        return "bird"
    low = name.lower()
    for word, visitor in (("squirrel", "squirrel"), ("raccoon", "raccoon"), ("opossum", "possum")):
        if word in low:
            return visitor
    return "other"


def pi_verdict(top: list[dict], local_share: float | None, check: dict | None) -> str:
    """sure: a strong first guess mostly on local species, and the animal check doesn't say empty;
    empty: the animal check says empty (or there's no guess) and the guess is weak; else unsure."""
    strong = bool(top) and top[0]["score"] >= SURE and (local_share is None or local_share >= classifier.GATE_MIN_LOCAL)
    empty = bool(check and check.get("score") is not None and check.get("empty"))
    if not top or (empty and not strong):
        return "empty"
    return "sure" if strong and not empty else "unsure"


def pi_say(verdict: str, top: list[dict]) -> str:
    if verdict == "empty":
        return "The Pi didn't find an animal here."
    if verdict == "sure":
        return f"Looks like {top[0]['name']}."
    return f"Maybe {top[0]['name']}, but the Pi isn't sure."


def ask_pi(conn, jpeg: bytes, circle: dict, night: bool, clf=None, local=None) -> dict:
    """The Pi's look at the circle: top 3 local species, the animal check's score and a verdict. Raises when the
    model can't be had (the route says so)."""
    import cv2

    global _last_used
    with _lock:
        _last_used = time.monotonic()
        clf = clf or classifier.load()
    if clf is None:
        raise RuntimeError("the local classifier is off (CLASSIFIER=off)")
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    start = time.perf_counter()
    patch, _ = classifier.crop_for(img, circle, None, MARGIN)
    emb = clf.embed(patch)
    guess = classifier.from_embedding(emb, night, local=local, clf=clf)
    check = presence.check(conn, emb)
    mammals = getattr(clf, "mammals", set())
    top = [{**g, "visitor": visitor_of(g["name"], g["scientific"], mammals)} for g in guess["top"]]
    verdict = pi_verdict(top, guess.get("local_share"), check)
    with _lock:
        _last_used = time.monotonic()
    return {"source": "pi", "model": classifier.MODEL_NAME, "verdict": verdict, "say": pi_say(verdict, top),
            "top": top, "local_share": guess.get("local_share"),
            "presence": None if not check or check.get("score") is None else
            {k: check[k] for k in ("score", "threshold", "empty")},
            "night": bool(night), "seconds": round(time.perf_counter() - start, 2)}


def cost(model_key: str, usage: dict) -> float:
    price = db.USD_PER_MTOK[model_key]
    return (usage.get("input_tokens", 0) * price["input"] + usage.get("output_tokens", 0) * price["output"]) / 1e6


def ask_claude(conn, source: str, jpeg: bytes, circle: dict, captured_at: datetime, size: tuple[int, int]) -> dict:
    """Haiku or Sonnet on the circle's square, with the same prompt as every identification. A bird's name is
    put in eBird's spelling when it has one."""
    box = square(circle, *size)
    sighting, meta = identify.ask(MODELS[source], [identify.prepare(jpeg, box)], captured_at,
                                  db.recent_lessons(conn), db.prompt_site(conn), note=NOTE)
    usage = {"input_tokens": meta.get("input_tokens", 0), "output_tokens": meta.get("output_tokens", 0)}
    entry = {"source": source, "model": meta.get("model") or MODELS[source], **usage,
             "usd": round(cost(source, usage), 6)}
    who = source.capitalize()
    if sighting is None:
        return {**entry, "verdict": "refused", "say": f"{who} declined to answer.", "answer": None}
    answer = sighting.model_dump(include={"visitor", "species", "count", "confidence", "notes"})
    name = sighting.species or ""
    if sighting.visitor == "bird" and name:
        name = names.canonical(name, [n for n, _ in db.species_kinds(conn)]) or name
    if sighting.visitor == "none":
        verdict, say = "empty", f"{who} didn't find an animal here."
    else:
        label = name or sighting.visitor.capitalize()
        verdict = "sure" if sighting.confidence == "high" else "unsure"
        say = f"{who} says {label} ({sighting.confidence})."
    return {**entry, "verdict": verdict, "say": say, "answer": answer, "name": name or None,
            "visitor": sighting.visitor}


def record(conn, visit_id: int, entry: dict, circle: dict) -> dict:
    """Keep the check with the visit (raw_json.calls.owner_checks, the newest KEEP). Returns what was stored."""
    row = conn.execute("SELECT raw_json FROM visits WHERE id = ?", (visit_id,)).fetchone()
    try:
        raw = json.loads(row["raw_json"] or "{}") or {}
    except ValueError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    calls = raw.setdefault("calls", {})
    stored = {**entry, "circle": {k: circle[k] for k in ("x", "y", "r")},
              "at": datetime.now().replace(microsecond=0).isoformat()}
    calls["owner_checks"] = [*(calls.get("owner_checks") or []), stored][-KEEP:]
    db.set_raw_json(conn, visit_id, raw)
    return stored


def frame_size(jpeg: bytes) -> tuple[int, int]:
    import io

    from PIL import Image

    return Image.open(io.BytesIO(jpeg)).size


def jpeg_of(visit) -> bytes | None:
    if visit["photo_deleted_at"]:
        return None
    path = config.IMAGES_DIR / visit["image_path"]
    return path.read_bytes() if path.exists() else None
