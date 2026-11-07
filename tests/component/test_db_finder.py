"""Component tests of db.finder_report: the finder chain's scorecard on a temp database."""
from datetime import timedelta

from helpers.clock import FROZEN_NOW
from helpers.finder import finder_raw
from birdbrain import classifier, db


def test_finder_report_counts_looks_none_and_named_per_finder(conn, add_visit):
    add_visit(model=classifier.MODEL_NAME, raw_json=finder_raw("blob", named=True))   # blob: a look, named
    add_visit(species="Bushtit", raw_json=finder_raw("motion"))                       # motion: a look, Claude named
    add_visit(model=classifier.MODEL_NAME, raw_json=finder_raw("yolox", named=True))  # yolox: a look, named
    add_visit(species="Bushtit", raw_json=finder_raw(None))                           # no patch: no look
    add_visit(species="Bushtit")                                                      # no finder record: not counted
    old = (FROZEN_NOW - timedelta(days=20)).isoformat()
    add_visit(model=classifier.MODEL_NAME, raw_json=finder_raw("blob", named=True), captured_at=old)  # too old
    report = db.finder_report(conn, days=14)
    assert report["looks"] == 3 and report["none"] == 1
    assert report["by_finder"]["blob"] == {"looks": 1, "named": 1}
    assert report["by_finder"]["motion"] == {"looks": 1, "named": 0}
    assert report["by_finder"]["yolox"] == {"looks": 1, "named": 1}
    wider = db.finder_report(conn, days=30)  # the old visit counts now
    assert wider["looks"] == 4 and wider["none"] == 1 and wider["by_finder"]["blob"] == {"looks": 2, "named": 2}


def test_finder_report_named_means_the_tight_look_named_the_visit(conn, add_visit):
    # A tight look that missed, with Claude's answer: the visit's model is Claude's even though a tight guess
    # (crop motion) is stored under classifier_tight. Not named.
    add_visit(species="Bushtit", raw_json=finder_raw("blob", score=0.9))
    # A tight look that named it, then corrected by the owner: still named by the look (the model stays bioclip-2).
    add_visit(model=classifier.MODEL_NAME, corrected_species="House Finch", raw_json=finder_raw("blob", named=True))
    report = db.finder_report(conn)
    assert report["looks"] == 2 and report["by_finder"]["blob"] == {"looks": 2, "named": 1}


def test_finder_report_on_an_empty_database(conn, add_visit):
    report = db.finder_report(conn)
    assert report["looks"] == 0 and report["none"] == 0
    assert not any(v["looks"] for v in report["by_finder"].values())
    add_visit(species="Bushtit")  # a visit from before the finder chain: still nothing to count
    report = db.finder_report(conn, days=14)
    assert report["looks"] == 0 and report["none"] == 0
