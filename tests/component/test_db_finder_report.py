"""Component tests of db.finder_report on a temp database: the builder's own, for what the test smith's in
test_db_finder.py do not pin (a look means a stored tight look, and the report's exact empty shape). The counts, the
days window and an empty database are pinned there."""

import json

from birdbrain import db

TIGHT = {"model": "bioclip-2", "crop": "motion", "top": [{"name": "Dark-eyed Junco", "score": 0.97}]}


def record(found, *tried):
    return {"finder": found, "tried": [{"finder": n, "found": n == found, "why": None if n == found else "nothing",
                                        "seconds": 0.1} for n in tried]}


def visit(add_visit, finder, named=False, bowl=False, **fields):
    calls = {"finder": finder}
    if named:
        calls.update(classifier={**TIGHT, "finder": finder["finder"]}, settled_by="classifier", skipped_claude=True)
        fields.setdefault("model", "bioclip-2")
    if bowl:
        calls.update(classifier={**TIGHT, "crop": "frame"}, settled_by="classifier")
        fields.setdefault("model", "bioclip-2")
    return add_visit(raw_json=json.dumps({"calls": calls}), **fields)


def test_a_look_means_a_stored_tight_look_not_just_a_finder_that_found_a_patch(conn, add_visit):
    visit(add_visit, record("motion", "motion"), named=True)
    visit(add_visit, record("blob", "motion", "blob"), bowl=True)  # the bowl look named it; the finder only circled it
    visit(add_visit, record("yolox", "motion", "blob", "yolox"))  # a patch, but the tight look failed: no look stored
    r = db.finder_report(conn)
    assert r["looks"] == 1 and r["none"] == 0 and r["by_finder"] == {"motion": {"looks": 1, "named": 1}}


def test_finder_report_has_the_exact_empty_shape(conn, add_visit):
    assert db.finder_report(conn) == {"days": 14, "looks": 0, "none": 0, "by_finder": {}}
    assert db.finder_report(conn, days=30)["days"] == 30
