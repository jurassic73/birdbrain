"""Component tests of Pi only (birdbrain/pi_only.py) in the watcher and its Stats scorecard (db.pi_only_report), on a
temp database (gate_yard): shadow stores the Pi's own verdict beside Claude's answer, "on" acts on it and never
asks Claude, "off" does neither. The camera, BioCLIP 2, the motion circle and Claude are fakes."""
import json
from datetime import timedelta

import pytest

from helpers.classifier import MOTION_CIRCLE, gate_yard, one_hot, sure  # noqa: F401  (a fixture)
from helpers.clock import FROZEN_NOW
from birdbrain import config, db, watch

KEYS = {"frames", "animals", "named", "named_wrong", "named_empty", "held_real", "held_empty", "missed",
        "claude_named", "empty"}


def latest(conn):
    row = conn.execute("SELECT * FROM visits ORDER BY id DESC").fetchone()
    return row, json.loads(row["raw_json"])["calls"]


def score(monkeypatch, value):
    """The Pi animal check's score for the next frames (gate_yard's is 0.9)."""
    monkeypatch.setattr(watch.presence, "check", lambda conn, e, view=None: {
        "model": "bioclip-presence", "score": value, "threshold": 0.02, "empty": False})


# --- shadow (the default): Claude is asked as ever, and the Pi's own verdict rides along -----------------------

def test_shadow_stores_what_the_pi_would_have_named_and_still_asks_claude(gate_yard, conn):
    assert config.PI_ONLY == "shadow"
    gate_yard["tight"] = sure("Dark-eyed Junco", 0.85)  # under the gate's 90%, over Pi only's 75%
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    row, calls = latest(conn)
    assert (row["species"], row["model"]) == ("Bushtit", "claude-haiku-5-5")  # Claude's answer stands
    assert calls["pi_only"] == {"verdict": "named", "score": 0.9, "species": "Dark-eyed Junco", "visitor": "bird",
                                "by": "tight"}
    assert "settled_by" not in calls and "skipped_claude" not in calls and db.skipped_by(row["raw_json"]) is None


def test_shadow_records_held_and_empty_by_the_animal_checks_score(gate_yard, conn, monkeypatch):
    gate_yard["tight"] = sure("Dark-eyed Junco", 0.6)  # no name at either bar
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    assert latest(conn)[1]["pi_only"] == {"verdict": "held", "score": 0.9, "why": "score"}
    score(monkeypatch, 0.5)
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False, False]
    assert latest(conn)[1]["pi_only"] == {"verdict": "empty", "score": 0.5}


def test_a_visit_the_gate_names_itself_carries_no_verdict(gate_yard, conn):
    assert watch.handle_event() == "bird" and not gate_yard["asked"]  # the tight look is sure (0.97)
    row, calls = latest(conn)
    assert row["model"] == "bioclip-2" and "pi_only" not in calls


def test_off_records_nothing(gate_yard, conn, monkeypatch):
    monkeypatch.setattr(config, "PI_ONLY", "off")
    gate_yard["tight"] = sure("Dark-eyed Junco", 0.85)
    assert watch.handle_event() == "bird" and gate_yard["asked"] == [False]
    assert "pi_only" not in latest(conn)[1]


# --- on: the Pi decides alone -----------------------------------------------------------------------------------

@pytest.fixture
def alone(gate_yard, monkeypatch):
    monkeypatch.setattr(config, "PI_ONLY", "on")
    return gate_yard


def test_on_names_the_visit_at_the_lower_bar_without_claude(alone, conn):
    alone["tight"] = sure("Dark-eyed Junco", 0.85)
    assert watch.handle_event() == "bird" and not alone["asked"]
    row, calls = latest(conn)
    assert (row["species"], row["model"], row["needs_review"]) == ("Dark-eyed Junco", "bioclip-2", 0)
    assert calls["settled_by"] == "classifier" and calls["skipped_claude"] and calls["circle"] == MOTION_CIRCLE
    assert calls["classifier"]["crop"] == "motion" and calls["classifier"]["top"][0]["score"] == 0.85
    assert calls["pi_only"]["verdict"] == "named" and calls["pi_only"]["by"] == "tight"
    assert "classifier_tight" not in calls  # the tight look named it: it is calls.classifier
    assert db.valid_json(json.dumps(calls))


def test_on_holds_a_frame_the_animal_check_is_sure_of_for_the_owner(alone, conn):
    alone["tight"] = sure("Dark-eyed Junco", 0.6)
    watch.handle_event()
    assert not alone["asked"]
    row, calls = latest(conn)
    assert (row["visitor"], row["species"], row["model"], row["needs_review"]) == ("other", None, "bioclip-presence", 1)
    assert calls["settled_by"] == "pi_only" and calls["skipped_claude"]
    assert calls["pi_only"] == {"verdict": "held", "score": 0.9, "why": "score"}
    assert calls["classifier_tight"]["top"][0]["score"] == 0.6  # the looks that fell short stay on the visit page
    assert "classifier" not in calls and "haiku" not in calls   # no scorecard look: nobody drew a circle
    assert db.skipped_by(row["raw_json"]) == "pi_only"
    # hidden from visitors like any "other", and waiting in the review queue
    assert conn.execute(f"SELECT COUNT(*) FROM visits WHERE {db.PUBLIC}").fetchone()[0] == 3  # the fixture's juncos


def test_on_files_a_low_score_as_a_false_trigger(alone, conn, monkeypatch):
    alone["tight"] = sure("Dark-eyed Junco", 0.6)
    score(monkeypatch, 0.5)
    assert watch.handle_event() == "none" and not alone["asked"]
    row, calls = latest(conn)
    assert (row["visitor"], row["model"], row["needs_review"]) == ("none", "bioclip-presence", 0)
    assert calls["settled_by"] == "pi_only" and calls["pi_only"] == {"verdict": "empty", "score": 0.5}
    assert "Claude is switched off" in row["notes"]


def test_on_names_a_sure_first_look_that_only_the_crowd_stopped(alone, conn):
    alone["emb"], alone["animals"] = one_hot(0), 2  # the first look is sure of a junco; two animals in view
    assert watch.handle_event() == "bird" and not alone["asked"]
    row, calls = latest(conn)
    assert (row["species"], row["model"]) == ("Dark-eyed Junco", "bioclip-2")
    assert calls["pi_only"]["by"] == "bowl" and calls["gate_crowd"] == {"species": "Dark-eyed Junco", "animals": 2}


# --- the Stats scorecard ----------------------------------------------------------------------------------------

def raw(verdict, species=None, **calls):
    record = {"verdict": verdict, "score": 0.5, **({"species": species, "visitor": "bird", "by": "tight"} if species else {})}
    return json.dumps({"calls": {"pi_only": record, "haiku": {"answer": {"visitor": "bird"}}, **calls}})


def test_report_is_all_zeros_with_the_exact_key_set_on_an_empty_database(conn, add_visit):
    assert db.pi_only_report(conn) == dict.fromkeys(KEYS, 0)
    add_visit(species="Bushtit")  # a visit from before the verdict was recorded
    assert db.pi_only_report(conn, days=14) == dict.fromkeys(KEYS, 0)


def test_report_scores_the_pis_verdict_against_the_final_label(conn, add_visit):
    add_visit(species="Dark-eyed Junco", raw_json=raw("named", "Dark-eyed Junco"))      # 1. named, right
    add_visit(species="Bushtit", raw_json=raw("named", "Dark-eyed Junco"))              # 2. named, wrong
    add_visit(visitor="none", species=None, raw_json=raw("named", "Dark-eyed Junco"))   # 3. named on an empty frame
    v = add_visit(species="Bushtit", raw_json=raw("named", "Dark-eyed Junco"))          # 4. the owner's label wins
    db.correct_visit(conn, v, "bird", "Dark-eyed Junco", None)
    add_visit(visitor="squirrel", species=None, raw_json=raw("named", "Eastern Gray Squirrel"))  # 5. a bare type matches
    add_visit(species="Bushtit", raw_json=raw("held"))                                  # 6. held, an animal
    add_visit(visitor="none", species=None, raw_json=raw("held"))                       # 7. held, empty
    add_visit(species="Bushtit", raw_json=raw("empty"))                                 # 8. missed; Claude named it
    v = add_visit(visitor="none", species=None, raw_json=raw("empty"))                  # 9. missed; Claude missed it too
    db.correct_visit(conn, v, "squirrel", "Eastern Gray Squirrel", None)
    add_visit(visitor="none", species=None, raw_json=raw("empty"))                      # 10. empty, as the Pi said
    # not scored: a verdict that was acted on (Claude never answered), and a visit with no verdict
    add_visit(visitor="other", species=None, model="bioclip-presence",
              raw_json=raw("held", settled_by="pi_only", skipped_claude=True))
    add_visit(species="Bushtit", raw_json=json.dumps({"calls": {"haiku": {"answer": {"visitor": "bird"}}}}))
    # 11. 30 days old, for the days filter
    add_visit(species="Bushtit", captured_at=(FROZEN_NOW - timedelta(days=30)).isoformat(), raw_json=raw("held"))
    r = db.pi_only_report(conn)
    assert set(r) == KEYS
    assert r == {"frames": 11, "animals": 8, "named": 5, "named_wrong": 1, "named_empty": 1, "held_real": 2,
                 "held_empty": 1, "missed": 2, "claude_named": 1, "empty": 1}
    recent = db.pi_only_report(conn, days=14)
    assert (recent["frames"], recent["held_real"]) == (10, 1)
