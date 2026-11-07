"""Seed data for the shadow agreement bar (db.agreement_report and the Stats table "Tight look bars (shadow)"):
visits with a bowl look (raw_json.calls.classifier_gate) and a tight look (calls.classifier_tight for a miss, or
calls.classifier with crop "motion" when BioCLIP 2 named the visit). Shared by the db and web component tests."""
import json

from birdbrain import classifier

BARS = (0.90, 0.93, 0.95)


def look(name: str, score: float, local_share: float = 0.9, crop: str = "circle", **extra) -> dict:
    """A BioCLIP 2 guess as the watcher stores it: a two-entry top list led by `name`."""
    other = "Bushtit" if name != "Bushtit" else "Dark-eyed Junco"
    return {"model": classifier.MODEL_NAME, "crop": crop, "night": False, "seconds": 5.9, "local_share": local_share,
            "top": [{"name": name, "scientific": "", "score": score},
                    {"name": other, "scientific": "", "score": round(1 - score, 4)}], **extra}


def bowl(name: str, score: float = 0.6) -> dict:
    return look(name, score, crop="bowl", why="score")


def miss_raw(bowl_name: str, tight_name: str, score: float, local_share: float = 0.9, **calls) -> str:
    """Both looks, the tight one a miss (Claude was asked)."""
    return json.dumps({"calls": {
        "classifier_gate": bowl(bowl_name),
        "classifier_tight": look(tight_name, score, local_share, crop=classifier.TIGHT_CROP, why="score",
                                 circle={"x": 0.48, "y": 0.5, "r": 0.04, "pi": True}),
        "haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}, **calls}})


def named_raw(bowl_name: str, tight_name: str, score: float, local_share: float = 0.9) -> str:
    """Both looks, the tight one named the visit (model bioclip-2, Claude not asked)."""
    return json.dumps({"calls": {
        "classifier_gate": bowl(bowl_name),
        "classifier": look(tight_name, score, local_share, crop=classifier.TIGHT_CROP,
                           circle={"x": 0.48, "y": 0.5, "r": 0.04, "pi": True}),
        "skipped_claude": True, "settled_by": "classifier"}})


def bowl_only_raw(bowl_name: str, scorecard: str | None = None) -> str:
    """A bowl look and no tight look; optionally the scorecard look made on Claude's circle afterwards (crop
    "circle"), which is not a tight look."""
    calls = {"classifier_gate": bowl(bowl_name),
             "haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}}
    if scorecard:
        calls["classifier"] = look(scorecard, 0.5)
    return json.dumps({"calls": calls})


def seed(add_visit) -> dict[str, int]:
    """The contract's seed, all daytime birds. Returns the visit ids by letter.

    A  tight miss 0.92, local 0.9, agrees with the bowl look, matches Claude's label (Dark-eyed Junco)
    B  tight miss 0.96, local 0.9, disagrees with the bowl look, wrong against the label (Bushtit)
    C  Pi-named tight visit 0.97, unconfirmed and uncorrected: excluded
    D  the same as C, confirmed_at set and matching the label: included
    E  only a bowl look: excluded
    F  tight miss 0.96 with local_share 0.3, agrees and matches: counted as a look but never named
    G  tight miss 0.94, local 0.9, agrees with the bowl look but both wrong against the label (Bushtit)

    Looks: A, B, D, F, G = 5.
      bar 0.90: named A B D G = 4, wrong B G = 2, agree_named A D G = 3, agree_wrong G = 1
      bar 0.93: named B D G = 3,   wrong B G = 2, agree_named D G = 2,   agree_wrong G = 1
      bar 0.95: named B D = 2,     wrong B = 1,   agree_named D = 1,     agree_wrong = 0
    """
    ids = {}
    ids["A"] = add_visit(species="Dark-eyed Junco", raw_json=miss_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.92))
    ids["B"] = add_visit(species="Bushtit", raw_json=miss_raw("House Finch", "Purple Finch", 0.96))
    ids["C"] = add_visit(species="Dark-eyed Junco", model=classifier.MODEL_NAME,
                         raw_json=named_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.97))
    ids["D"] = add_visit(species="Dark-eyed Junco", model=classifier.MODEL_NAME, confirmed_at="2026-11-06T12:30:00",
                         raw_json=named_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.97))
    ids["E"] = add_visit(species="Bushtit", raw_json=bowl_only_raw("Dark-eyed Junco"))
    ids["F"] = add_visit(species="Dark-eyed Junco",
                         raw_json=miss_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.96, local_share=0.3))
    ids["G"] = add_visit(species="Bushtit", raw_json=miss_raw("House Finch", "House Finch", 0.94))
    return ids


EXPECTED = {
    "looks": 5,
    "bars": [
        {"bar": 0.90, "named": 4, "wrong": 2, "agree_named": 3, "agree_wrong": 1},
        {"bar": 0.93, "named": 3, "wrong": 2, "agree_named": 2, "agree_wrong": 1},
        {"bar": 0.95, "named": 2, "wrong": 1, "agree_named": 1, "agree_wrong": 0},
    ],
}
