"""Component tests of db.finder_report: how the finder chain for the BioCLIP 2 tight look did, on a temp database."""

import json
from datetime import timedelta

from helpers.clock import FROZEN_NOW
from birdbrain import db

TIGHT = {"model": "bioclip-2", "crop": "motion", "top": [{"name": "Dark-eyed Junco", "score": 0.97}]}


def record(found, *tried):
    return {"finder": found, "tried": [{"finder": n, "found": n == found, "why": None if n == found else "nothing",
                                        "seconds": 0.1} for n in tried]}


def visit(add_visit, finder, named=False, miss=False, bowl=False, **fields):
    calls = {"finder": finder}
    if named:
        calls.update(classifier={**TIGHT, "finder": finder["finder"]}, settled_by="classifier", skipped_claude=True)
        fields.setdefault("model", "bioclip-2")
    if miss:
        calls.update(classifier_tight={**TIGHT, "finder": finder["finder"], "why": "score"})
    if bowl:
        calls.update(classifier={**TIGHT, "crop": "frame"}, settled_by="classifier")
        fields.setdefault("model", "bioclip-2")
    return add_visit(raw_json=json.dumps({"calls": calls}), **fields)


def test_finder_report_counts_looks_none_and_by_finder(conn, add_visit):
    visit(add_visit, record("motion", "motion"), named=True)
    visit(add_visit, record("motion", "motion"), miss=True)
    visit(add_visit, record("blob", "motion", "blob"), named=True)
    visit(add_visit, record("yolox", "motion", "blob", "yolox"), miss=True)
    visit(add_visit, record(None, "motion", "blob", "yolox"))  # no patch: Claude was asked
    visit(add_visit, record(None, "motion"))
    visit(add_visit, record("blob", "motion", "blob"), bowl=True)  # the bowl look named it; the finder only circled it
    visit(add_visit, record("yolox", "motion", "blob", "yolox"))  # a patch, but the tight look failed: no look stored
    add_visit()  # an old visit without a finder record
    r = db.finder_report(conn)
    assert r["days"] == 14 and r["looks"] == 4 and r["none"] == 2
    assert r["by_finder"] == {"motion": {"looks": 2, "named": 1}, "blob": {"looks": 1, "named": 1},
                              "yolox": {"looks": 1, "named": 0}}


def test_finder_report_is_empty_without_records_and_honours_days(conn, add_visit):
    assert db.finder_report(conn) == {"days": 14, "looks": 0, "none": 0, "by_finder": {}}
    old = (FROZEN_NOW - timedelta(days=20)).isoformat()
    visit(add_visit, record("motion", "motion"), named=True, captured_at=old)
    visit(add_visit, record(None, "motion"), captured_at=old)
    assert db.finder_report(conn)["looks"] == 0 and db.finder_report(conn)["none"] == 0
    assert db.finder_report(conn, days=30) == {"days": 30, "looks": 1, "none": 1, "by_finder": {"motion": {"looks": 1, "named": 1}}}
    assert db.finder_report(conn, days=None)["looks"] == 1
