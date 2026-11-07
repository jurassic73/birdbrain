"""Shared helpers for the finder chain tests (birdbrain/finder.py, used by more than one test module): the crowd
counter's box and the circles the finders give, the motion attempt that missed, and a visit's stored raw_json with
a finder record."""
import json

from helpers.classifier import MOTION_CIRCLE, MOTION_FOUND, guess
from birdbrain import classifier

BOX = [0.4, 0.5, 0.1, 0.2]                                    # the crowd counter's box: x, y, w, h frame fractions
BLOB_CIRCLE = {"x": 0.45, "y": 0.6, "r": 0.0562, "pi": True}  # box_circle(BOX): the centre, r = max(w, h * 9/16) / 2
YOLOX_BOX = [0.3, 0.3, 0.1, 0.1]                              # detector.animal_box's box, full-frame fractions
YOLOX_CIRCLE = {"x": 0.35, "y": 0.35, "r": 0.05, "pi": True}  # box_circle(YOLOX_BOX)
# motion_circle.live's record when it found no patch (the shape it stores today)
MISS = {"found": False, "why": "nothing", "source": "empty", "threshold": 25.0, "changed": 0.012, "where": "land",
        "seconds": 0.02}
CIRCLES = {"motion": MOTION_CIRCLE, "blob": BLOB_CIRCLE, "yolox": YOLOX_CIRCLE}
WHY = {"motion": "two alike", "blob": "no box", "yolox": "nothing"}  # each finder's miss, in a record with none found


def tried(found):
    """The chain's tried list, in FINDERS order, when `found` (a finder name, or None) found the patch."""
    out = []
    for name in ("motion", "blob", "yolox"):
        hit = name == found
        out.append({"finder": name, "found": hit, "why": None if hit else WHY[name], "seconds": 0.1})
        if hit:
            break
    return out


def record(found):
    """raw_json.finder as the watcher stores it."""
    return {"finder": found, "tried": tried(found), "motion": MOTION_FOUND if found == "motion" else MISS}


def finder_raw(found, named=False, score=0.97, name="Dark-eyed Junco"):
    """A visit's raw_json with a first look on the bowl that fell short and a finder record. `found` None: no finder
    found a patch, so there was no tight look and Claude answered. Otherwise the tight look was made on the circle
    that finder gave: `named`, it named the visit (calls.classifier, crop motion, the visit's model is bioclip-2);
    else it missed (calls.classifier_tight) and Claude answered, with the scorecard look as calls.classifier."""
    first = {"model": classifier.MODEL_NAME, "crop": "bowl", "night": False, "seconds": 0.0, "why": "score",
             "local_share": 0.9, "top": [{"name": "Black-capped Chickadee", "scientific": "", "score": 0.61},
                                         {"name": "Bushtit", "scientific": "", "score": 0.2}]}
    claude = {"haiku": {"frames": 1, "answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}}
    calls = {"finder": record(found), "classifier_gate": first}
    if found is None:
        calls.update(claude, classifier=guess("Bushtit"), motion_circle=MISS)
        return json.dumps({"calls": calls})
    tight = {"model": classifier.MODEL_NAME, "crop": classifier.TIGHT_CROP, "night": False, "seconds": 5.9,
             "local_share": 0.9, "finder": found,
             "top": [{"name": name, "scientific": "", "score": score}, {"name": "Bushtit", "scientific": "", "score": 0.02}]}
    if named:
        calls.update(classifier=tight, skipped_claude=True, settled_by="classifier", circle=CIRCLES[found],
                     motion_circle=record(found)["motion"])
    else:
        calls.update(claude, classifier=guess("Bushtit"),
                     classifier_tight={**tight, "why": "score", "circle": CIRCLES[found]})
        if found != "motion":
            calls["motion_circle"] = MISS  # the motion attempt missed: kept, as today
    return json.dumps({"calls": calls})
