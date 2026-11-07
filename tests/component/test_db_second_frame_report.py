"""Component tests of db.second_frame_report (the shadow second-frame look's scorecard) on a temp database: how
often the tight look's first frame, the second frame and their averaged embedding put the final label first, and
how many visits the second frame or the average would have named, and wrongly."""
import json

from helpers.agreement import look
from birdbrain import classifier, db

KEYS = {"looks", "first_right", "second_right", "combined_right", "combined_named", "combined_wrong", "second_named",
        "second_wrong"}


def second(name, score, combined_name=None, combined_score=None, local_share=0.9):
    rec = {**look(name, score, crop=classifier.TIGHT_CROP), "frame": "second", "seconds": 6.0, "combined": None}
    if combined_name:
        rec["combined"] = {"top": [{"name": combined_name, "scientific": "", "score": combined_score}],
                           "local_share": local_share}
    return rec


def raw(tight_name, tight_score, second_rec, **calls):
    return json.dumps({"calls": {
        "classifier_tight": look(tight_name, tight_score, crop=classifier.TIGHT_CROP, why="score"),
        "classifier_second": second_rec,
        "haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}, **calls}})


def seed(add_visit):
    """Six visits with a second-frame look, all daytime birds, labelled Dark-eyed Junco unless said otherwise.

    A  first right 0.9; second right 0.96 (named, right); averaged right 0.97 (named, right)
    B  first wrong (Bushtit 0.85); second right 0.88 (not named); averaged right 0.95 (named, right)
    C  first right 0.9; second wrong (Bushtit 0.96: named, wrong); averaged wrong (Bushtit 0.96: named, wrong)
    D  first right 0.9; second right 0.9; no combined (no tight embedding)
    E  first right 0.9; second right 0.97 but local_share 0.3 (never named); averaged right 0.97, local 0.3 (never named)
    F  a Pi-named visit (model bioclip-2) with a second look, unconfirmed: excluded
    G  a visit labelled only "bird" (no species): excluded
    """
    add_visit(raw_json=raw("Dark-eyed Junco", 0.9, second("Dark-eyed Junco", 0.96, "Dark-eyed Junco", 0.97)))
    add_visit(raw_json=raw("Bushtit", 0.85, second("Dark-eyed Junco", 0.88, "Dark-eyed Junco", 0.95)))
    add_visit(raw_json=raw("Dark-eyed Junco", 0.9, second("Bushtit", 0.96, "Bushtit", 0.96)))
    add_visit(raw_json=raw("Dark-eyed Junco", 0.9, second("Dark-eyed Junco", 0.9)))
    weak = second("Dark-eyed Junco", 0.97, "Dark-eyed Junco", 0.97, local_share=0.3)
    weak["local_share"] = 0.3
    add_visit(raw_json=raw("Dark-eyed Junco", 0.9, weak))
    add_visit(model=classifier.MODEL_NAME, raw_json=raw("Dark-eyed Junco", 0.9, second("Dark-eyed Junco", 0.96)))
    add_visit(species=None, raw_json=raw("Dark-eyed Junco", 0.9, second("Dark-eyed Junco", 0.96)))


def test_counts_on_the_seeded_yard(conn, add_visit):
    seed(add_visit)
    report = db.second_frame_report(conn)
    assert set(report) == KEYS
    assert report == {"looks": 5, "first_right": 4, "second_right": 4, "combined_right": 3,
                      "second_named": 2, "second_wrong": 1, "combined_named": 3, "combined_wrong": 1}


def test_an_owner_correction_is_the_label_and_a_checked_pi_visit_counts(conn, add_visit):
    vid = add_visit(species="Bushtit", raw_json=raw("Dark-eyed Junco", 0.9, second("Dark-eyed Junco", 0.96, "Dark-eyed Junco", 0.97)))
    assert db.second_frame_report(conn)["first_right"] == 0
    conn.execute("UPDATE visits SET corrected_species = 'Dark-eyed Junco' WHERE id = ?", (vid,))
    assert db.second_frame_report(conn)["first_right"] == 1
    add_visit(model=classifier.MODEL_NAME, confirmed_at="2026-11-06T12:30:00",
              raw_json=raw("Dark-eyed Junco", 0.9, second("Dark-eyed Junco", 0.96)))
    assert db.second_frame_report(conn)["looks"] == 2

