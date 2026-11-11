"""Where the animal is, for the BioCLIP 2 species check's tight look (rule H in watch.py).

The tight look is right far more often than the wide look at the whole bowl, but it needs a patch to look at. The
Motion detector's circle (motion_circle.py) finds one on about two visits in three; this module tries a chain of
finders, in `config.FINDERS` order, until one points at the animal:

- "motion": the Motion detector's circle, the biggest patch of change against the empty scene (as before).
- "blob": the crowd counter's outline (stay.count_animals' box per counted animal): exactly one box is the animal.
  Costs nothing; the boxes were already made to count the animals.
- "yolox": a small object detector (YOLOX-tiny, detector.py) on the crop box, by day only: the best bird, cat, dog
  or other animal-class detection. Its class-agnostic "objects" land on rocks and leaves and are never used.

Every finder is fail-soft: a finder that fails is recorded and the next one runs, and `locate` never raises. What
each finder found or why not is kept with the visit (raw_json.finder) and shown on the visit page and Stats.
"""
from __future__ import annotations

import logging
import time
from decimal import ROUND_HALF_EVEN, Decimal

from . import config, detector, motion_circle

log = logging.getLogger("birdbrain.finder")

FINDERS = ("motion", "blob", "yolox")  # the chain, in order; config.FINDERS says which run and in what order
NIGHT_SKIPPED = ("blob", "yolox")  # at night only the Motion detector runs (YOLOX was trained on daylight photos)


def _round4(v: float) -> float:
    """Four places, rounding a half to even on the decimal value (0.05625 is 0.0562), where round() would go by the
    float's binary error."""
    return float(Decimal(repr(float(v))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN))


def box_circle(box, aspect: float = 9 / 16) -> dict:
    """A circle like the Motion detector's ({x, y, r, pi}, frame fractions, r as a fraction of the width) around an
    [x, y, w, h] frame-fraction box: the box's centre, reaching its longer side (the height scaled by the frame's
    aspect so it is in width units), at least 0.01."""
    x, y, w, h = (float(v) for v in box[:4])
    r = max(w, h * aspect) / 2
    return {"x": _round4(x + w / 2), "y": _round4(y + h / 2), "r": _round4(max(r, 0.01)), "pi": True}


def _motion(jpeg, empty, z, trigger, captured_at, night, record) -> tuple[dict | None, str | None]:
    circle, info = motion_circle.live(jpeg, empty, z, trigger, captured_at, night)
    record["motion"] = info
    return circle, None if circle else (info or {}).get("why") or "nothing"


def _blob(boxes) -> tuple[dict | None, str | None]:
    if boxes is None:
        return None, "no count"
    if len(boxes) == 0:
        return None, "no box"
    if len(boxes) > 1:
        return None, "several"
    circle = box_circle(boxes[0])
    if circle["r"] > config.CIRCLE_MAX_R:
        # The counter outlined most of the crop (a cloud or a glint changed it): the light, not an animal, by the
        # same bar the Motion detector uses for its patches. A tight look on it would be a look at the whole bowl.
        return None, "too big"
    return circle, None


def _yolox(jpeg, crop, attempt: dict) -> tuple[dict | None, str | None]:
    box, info = detector.animal_box(jpeg, crop, min_score=config.FINDER_YOLOX_MIN_SCORE)
    for key in ("label", "score"):
        if info.get(key) is not None:
            attempt[key] = info[key]
    if box is None:
        return None, info.get("why") or "nothing"
    return box_circle(box), None


def locate(jpeg: bytes, empty, z, trigger: str, captured_at: str, night: bool, boxes: list | None = None,
           crop=None) -> tuple[dict | None, dict]:
    """The circle around the animal from the first finder in `config.FINDERS` that finds one, and the record of the
    chain: {"finder": the name that found it or None, "tried": [{finder, found, why, seconds}, ...], "motion": the
    Motion detector's info dict when it ran}. Never raises."""
    record: dict = {"finder": None, "tried": []}
    for name in config.FINDERS:
        attempt: dict = {"finder": name, "found": False, "why": None, "seconds": 0.0}  # no timing for a skipped finder
        circle = None
        if night and name in NIGHT_SKIPPED:
            attempt["why"] = "night"
        else:
            started = time.perf_counter()
            try:
                if name == "motion":
                    circle, attempt["why"] = _motion(jpeg, empty, z, trigger, captured_at, night, record)
                elif name == "blob":
                    circle, attempt["why"] = _blob(boxes)
                elif name == "yolox":
                    circle, attempt["why"] = _yolox(jpeg, crop, attempt)
                else:
                    attempt["why"] = "unknown"
            except Exception:
                log.exception("the %s finder failed; trying the next", name)
                circle, attempt["why"] = None, "error"
                if name == "motion" and "motion" not in record:
                    # It ran and failed: say so (stored as calls.motion_circle), not that FINDERS left it out.
                    record["motion"] = {"found": False, "why": "error", "source": None}
            attempt["seconds"] = round(time.perf_counter() - started, 3)
        attempt["found"] = circle is not None
        record["tried"].append(attempt)
        if circle is not None:
            record["finder"] = name
            return circle, record
    return None, record


# What each finder says when it looked properly and saw nothing: not skipped, failed, left out of FINDERS or blinded
# (the Motion detector's "light", "too big" and "two alike" all mean something did change).
CLEAR = {"motion": "nothing", "blob": "no box", "yolox": "nothing"}


def nothing_on_land(record: dict | None) -> bool:
    """Whether the whole chain looked and every finder saw nothing new on land: no patch of change against the
    trigger's own empty scene (nor one compact patch on the water, a bathing bird), no animal outlined by the crowd
    counter, none seen by the object detector. False whenever a finder was skipped, failed or left out: then nobody
    really looked."""
    if not isinstance(record, dict) or record.get("finder"):
        return False
    tried = {t.get("finder"): t for t in record.get("tried") or [] if isinstance(t, dict)}
    if set(tried) != set(CLEAR) or (record.get("motion") or {}).get("source") != "empty":
        return False
    return all(not t.get("found") and t.get("why") == CLEAR[name] for name, t in tried.items())


def tried_text(record: dict | None) -> str:
    """One line for a log or a tip: each finder tried and why it found nothing ("motion: nothing, blob: no box")."""
    tried = (record or {}).get("tried") or []
    return ", ".join(f"{t.get('finder')}: {'found' if t.get('found') else t.get('why') or 'nothing'}" for t in tried) or "none"
