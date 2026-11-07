"""Shared helpers for the shadow second-frame look tests (used by more than one test module): a visit's raw_json
with a first look on the bowl, a tight look that fell short, and the second-frame look with or without its average."""
import json

from helpers.classifier import MOTION_CIRCLE
from birdbrain import classifier

CLAUDE = {"frames": 1, "answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}


def look(name: str, score: float, local_share: float = 0.9, crop: str = classifier.TIGHT_CROP) -> dict:
    """A one-species BioCLIP 2 guess as the watcher stores it."""
    return {"model": classifier.MODEL_NAME, "crop": crop, "night": False, "seconds": 5.9, "local_share": local_share,
            "top": [{"name": name, "scientific": "", "score": score}]}


def second_frame_raw(first, second, combined=None, with_second: bool = True, **calls) -> str:
    """raw_json for a visit Claude named after the tight look (`first`: (name, score[, local_share])) fell short and
    the shadow look at the next frame (`second`) was stored, averaged with the tight look when `combined` is given.
    `with_second=False` leaves the second-frame look out (a visit from before the shadow look)."""
    tight = {**look(*first), "why": "score", "circle": MOTION_CIRCLE}
    stored = {
        "classifier_gate": {**look("Black-capped Chickadee", 0.61, crop="bowl"), "why": "score"},
        "classifier_tight": tight,
        "haiku": CLAUDE,
        "classifier": look("Bushtit", 0.5, crop="circle"),
        **calls,
    }
    if with_second:
        avg = None
        if combined is not None:
            c = look(*combined)
            avg = {"top": c["top"], "local_share": c["local_share"]}
        stored["classifier_second"] = {**look(*second), "frame": "second", "combined": avg}
    return json.dumps({"calls": stored})
