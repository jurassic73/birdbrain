"""Component tests of db.agreement_report (the shadow agreement bar) on a temp database: how often the BioCLIP 2
tight look would have named a visit at a lower bar, how often wrongly, and the same when the bowl look agrees."""
import json

import pytest

from helpers.agreement import BARS, EXPECTED, bowl_only_raw, miss_raw, named_raw, seed
from helpers.clock import FROZEN_NOW
from birdbrain import classifier, db


def test_counts_at_each_bar_on_the_seeded_yard(conn, add_visit):
    seed(add_visit)
    report = db.agreement_report(conn, bars=BARS)
    assert report["looks"] == EXPECTED["looks"]
    assert [b["bar"] for b in report["bars"]] == list(BARS)
    assert report["bars"] == EXPECTED["bars"]


def test_default_bars_are_90_93_and_95_over_all_time(conn, add_visit):
    seed(add_visit)
    assert db.agreement_report(conn) == db.agreement_report(conn, bars=(0.90, 0.93, 0.95), days=None)
    assert [b["bar"] for b in db.agreement_report(conn)["bars"]] == [0.90, 0.93, 0.95]


def test_empty_database_gives_zero_looks_and_zero_rows(conn):
    report = db.agreement_report(conn, bars=BARS)
    assert report["looks"] == 0
    assert report["bars"] == [{"bar": b, "named": 0, "wrong": 0, "agree_named": 0, "agree_wrong": 0} for b in BARS]


def test_a_pi_named_tight_visit_counts_only_once_confirmed_or_corrected(conn, add_visit):
    # Unconfirmed and uncorrected: BioCLIP 2 would be scoring its own answer, so it is left out.
    vid = add_visit(species="Dark-eyed Junco", model=classifier.MODEL_NAME,
                    raw_json=named_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.97))
    assert db.agreement_report(conn, bars=BARS)["looks"] == 0
    # Confirmed by the owner: included, right at every bar, and the bowl look agrees.
    conn.execute("UPDATE visits SET confirmed_at = ? WHERE id = ?", (FROZEN_NOW.isoformat(), vid))
    conn.commit()
    report = db.agreement_report(conn, bars=BARS)
    assert report["looks"] == 1
    assert report["bars"] == [{"bar": b, "named": 1, "wrong": 0, "agree_named": 1, "agree_wrong": 0} for b in BARS]


def test_a_pi_named_tight_visit_the_owner_corrected_counts_as_wrong(conn, add_visit):
    # Pi said Dark-eyed Junco at 0.97 and the bowl look agreed; the owner corrected it to a Bushtit.
    add_visit(species="Dark-eyed Junco", model=classifier.MODEL_NAME, corrected_species="Bushtit",
              raw_json=named_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.97))
    report = db.agreement_report(conn, bars=BARS)
    assert report["looks"] == 1
    assert report["bars"] == [{"bar": b, "named": 1, "wrong": 1, "agree_named": 1, "agree_wrong": 1} for b in BARS]


def test_the_owners_correction_wins_over_claudes_label(conn, add_visit):
    # Claude said Bushtit, the tight look said Purple Finch at 0.96: wrong against Claude, right once corrected.
    vid = add_visit(species="Bushtit", raw_json=miss_raw("Purple Finch", "Purple Finch", 0.96))
    assert db.agreement_report(conn, bars=BARS)["bars"][0] == {"bar": 0.90, "named": 1, "wrong": 1,
                                                               "agree_named": 1, "agree_wrong": 1}
    conn.execute("UPDATE visits SET corrected_species = 'Purple Finch' WHERE id = ?", (vid,))
    conn.commit()
    assert db.agreement_report(conn, bars=BARS)["bars"][0] == {"bar": 0.90, "named": 1, "wrong": 0,
                                                               "agree_named": 1, "agree_wrong": 0}


def test_a_visit_with_only_a_bowl_look_is_not_a_look(conn, add_visit):
    add_visit(species="Bushtit", raw_json=bowl_only_raw("Dark-eyed Junco"))
    # The scorecard look on Claude's circle afterwards (crop "circle") is not the tight look either.
    add_visit(species="Bushtit", raw_json=bowl_only_raw("Dark-eyed Junco", scorecard="Bushtit"))
    report = db.agreement_report(conn, bars=BARS)
    assert report["looks"] == 0 and all(b["named"] == 0 for b in report["bars"])


def test_a_tight_look_without_a_bowl_look_is_not_a_look(conn, add_visit):
    raw = json.loads(miss_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.96))
    del raw["calls"]["classifier_gate"]
    add_visit(species="Dark-eyed Junco", raw_json=json.dumps(raw))
    assert db.agreement_report(conn, bars=BARS)["looks"] == 0


def test_low_local_share_is_never_named_at_any_bar(conn, add_visit):
    add_visit(species="Dark-eyed Junco",
              raw_json=miss_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.99, local_share=classifier.GATE_MIN_LOCAL - 0.01))
    report = db.agreement_report(conn, bars=BARS)
    assert report["looks"] == 1
    assert report["bars"] == [{"bar": b, "named": 0, "wrong": 0, "agree_named": 0, "agree_wrong": 0} for b in BARS]
    # Exactly GATE_MIN_LOCAL (0.5) is enough, like the gate itself (gate_miss uses "<").
    add_visit(species="Dark-eyed Junco",
              raw_json=miss_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.99, local_share=classifier.GATE_MIN_LOCAL))
    assert db.agreement_report(conn, bars=BARS)["bars"][0]["named"] == 1


def test_a_score_exactly_at_the_bar_is_named(conn, add_visit):
    add_visit(species="Dark-eyed Junco", raw_json=miss_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.93))
    named = {b["bar"]: b["named"] for b in db.agreement_report(conn, bars=BARS)["bars"]}
    assert named == {0.90: 1, 0.93: 1, 0.95: 0}


def test_agreement_is_between_the_two_top_names_only(conn, add_visit):
    # The bowl look's second guess is the tight look's answer: that is not agreement.
    raw = json.loads(miss_raw("Bushtit", "Dark-eyed Junco", 0.96))
    assert raw["calls"]["classifier_gate"]["top"][1]["name"] == "Dark-eyed Junco"
    add_visit(species="Dark-eyed Junco", raw_json=json.dumps(raw))
    row = db.agreement_report(conn, bars=BARS)["bars"][0]
    assert row == {"bar": 0.90, "named": 1, "wrong": 0, "agree_named": 0, "agree_wrong": 0}


def test_names_match_like_the_scorecard_ignoring_case_and_punctuation(conn, add_visit):
    # classifier_report's _name_key matching: "dark eyed junco" is the label "Dark-eyed Junco".
    add_visit(species="Dark-eyed Junco", raw_json=miss_raw("DARK-EYED JUNCO", "dark eyed junco", 0.96))
    row = db.agreement_report(conn, bars=BARS)["bars"][0]
    assert row == {"bar": 0.90, "named": 1, "wrong": 0, "agree_named": 1, "agree_wrong": 0}


def test_a_generic_mammal_label_matches_a_species_ending_in_it(conn, add_visit):
    # Same rule as classifier_report: label "Raccoon" counts "Common Raccoon" as right.
    add_visit(visitor="raccoon", species="Raccoon", raw_json=miss_raw("Common Raccoon", "Common Raccoon", 0.96))
    row = db.agreement_report(conn, bars=BARS)["bars"][0]
    assert row == {"bar": 0.90, "named": 1, "wrong": 0, "agree_named": 1, "agree_wrong": 0}


def test_a_visit_without_a_species_label_is_left_out(conn, add_visit):
    # Both looks are there but Claude named no species (label "Bird"): nothing to judge the tight look against.
    add_visit(species=None, raw_json=miss_raw("Dark-eyed Junco", "Dark-eyed Junco", 0.96))
    assert db.agreement_report(conn, bars=BARS)["looks"] == 0


def test_days_limits_the_report_to_recent_visits(conn, add_visit):
    seed(add_visit)  # all captured at FROZEN_NOW
    add_visit(species="Bushtit", captured_at="2020-01-01T09:00:00",
              raw_json=miss_raw("Purple Finch", "Purple Finch", 0.96))  # long ago: wrong, agreeing
    assert db.agreement_report(conn, bars=BARS, days=None)["looks"] == EXPECTED["looks"] + 1
    recent = db.agreement_report(conn, bars=BARS, days=14)
    assert recent["looks"] == EXPECTED["looks"]
    assert recent["bars"] == EXPECTED["bars"]


def test_custom_bars_are_honoured_in_the_given_order(conn, add_visit):
    seed(add_visit)
    report = db.agreement_report(conn, bars=(0.99, 0.5))
    assert [b["bar"] for b in report["bars"]] == [0.99, 0.5]
    # 0.99: nothing reaches it. 0.5: every look with local_share >= 0.5 (A B D G; F has 0.3).
    assert report["bars"][0] == {"bar": 0.99, "named": 0, "wrong": 0, "agree_named": 0, "agree_wrong": 0}
    assert report["bars"][1] == {"bar": 0.5, "named": 4, "wrong": 2, "agree_named": 3, "agree_wrong": 1}
