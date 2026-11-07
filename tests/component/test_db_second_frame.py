"""Component tests of db.second_frame_report (the shadow second-frame look's scorecard) on a temp database."""

import json
from datetime import timedelta

from helpers.classifier import MOTION_CIRCLE
from helpers.clock import FROZEN_NOW
from helpers.second_frame import look, second_frame_raw
from birdbrain import classifier, db

KEYS = {"looks", "first_right", "second_right", "combined_right", "combined_named", "combined_wrong", "second_named",
        "second_wrong"}


def test_report_is_all_zeros_with_the_exact_key_set_on_an_empty_database(conn, add_visit):
    r = db.second_frame_report(conn)
    assert set(r) == KEYS and set(r.values()) == {0}
    # a tight look alone (no second frame) is not a look
    add_visit(species="Bushtit", raw_json=second_frame_raw(("Dark-eyed Junco", 0.85), None, with_second=False))
    assert db.second_frame_report(conn) == dict.fromkeys(KEYS, 0)
    assert db.second_frame_report(conn, days=14) == dict.fromkeys(KEYS, 0)


def test_report_scores_first_second_and_averaged_against_the_final_label(conn, add_visit):
    bar, local = classifier.GATE_MIN_SCORE_TIGHT, classifier.GATE_MIN_LOCAL
    assert (bar, local) == (0.95, 0.5)  # the seeds below sit on either side of these bars
    # 1. all three right; the second and the average would both have named it
    add_visit(species="Dark-eyed Junco", raw_json=second_frame_raw(("Dark-eyed Junco", 0.85), ("Dark-eyed Junco", 0.97),
                                                                   ("Dark-eyed Junco", 0.96)))
    # 2. all wrong; the second frame would have named it (wrongly), the average falls under the bar
    add_visit(species="Bushtit", raw_json=second_frame_raw(("Dark-eyed Junco", 0.9), ("Dark-eyed Junco", 0.96),
                                                           ("Dark-eyed Junco", 0.93)))
    # 3. the owner's label wins (case ignored); the average is right but too little of it is on local species
    v = add_visit(species="Bushtit", raw_json=second_frame_raw(("House Finch", 0.88), ("Bushtit", 0.5),
                                                               ("House Finch", 0.97, 0.3)))
    db.correct_visit(conn, v, "bird", "house finch", None)
    # 4. named by the Pi and never checked: not scored
    add_visit(model="bioclip-2", raw_json=second_frame_raw(("Dark-eyed Junco", 0.9), ("Dark-eyed Junco", 0.99)))
    # 5. the same, confirmed by the owner: scored (no average stored)
    v = add_visit(model="bioclip-2", raw_json=second_frame_raw(("Dark-eyed Junco", 0.9), ("Dark-eyed Junco", 0.99)))
    db.confirm_visits(conn, [v])
    # 6. "Bird" with no species, 7. no second-frame look, 8. a false trigger: none of them scored
    add_visit(species=None, raw_json=second_frame_raw(("Dark-eyed Junco", 0.9), ("Dark-eyed Junco", 0.99)))
    add_visit(species="Bushtit", raw_json=second_frame_raw(("Dark-eyed Junco", 0.85), None, with_second=False))
    add_visit(visitor="none", species=None, raw_json=second_frame_raw(("Dark-eyed Junco", 0.9), ("Dark-eyed Junco", 0.99)))
    # 9. a generic mammal label matches a species ending in it; 30 days old for the days filter
    add_visit(visitor="raccoon", species=None, captured_at=(FROZEN_NOW - timedelta(days=30)).isoformat(),
              raw_json=second_frame_raw(("Common Raccoon", 0.9), ("Common Raccoon", 0.96)))
    r = db.second_frame_report(conn)
    assert set(r) == KEYS
    assert r == {"looks": 5, "first_right": 4, "second_right": 3, "combined_right": 2, "combined_named": 1,
                 "combined_wrong": 0, "second_named": 4, "second_wrong": 1}
    recent = db.second_frame_report(conn, days=7)
    assert recent == {**r, "looks": 4, "first_right": 3, "second_right": 2, "second_named": 3}


def test_report_reads_a_second_look_stored_without_the_combined_key(conn, add_visit):
    """A record the watcher stored when the average could not be made still counts as a look."""
    second = {**look("Dark-eyed Junco", 0.6), "frame": "second"}  # no "combined" at all
    add_visit(species="Dark-eyed Junco", raw_json=json.dumps({"calls": {
        "classifier_tight": {**look("Bushtit", 0.9), "why": "score", "circle": MOTION_CIRCLE},
        "classifier_second": second,
        "haiku": {"answer": {"visitor": "bird", "species": "Dark-eyed Junco", "confidence": "high"}}}}))
    r = db.second_frame_report(conn)
    assert (r["looks"], r["first_right"], r["second_right"], r["second_named"]) == (1, 0, 1, 0)
    assert (r["combined_right"], r["combined_named"], r["combined_wrong"]) == (0, 0, 0)
