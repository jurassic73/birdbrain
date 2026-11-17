"""Unit tests of birdbrain/pi_only (what the Pi decides when Claude is not asked) and the two results it leads to
(identify.settled_by_pi_only). Pure: dicts in, dicts out."""
import pytest

from birdbrain import classifier, config, identify, pi_only

SETTLED = {"Dark-eyed Junco": "bird", "Eastern Gray Squirrel": "squirrel"}


def look(name, score, local_share=0.9, **extra):
    return {"model": classifier.MODEL_NAME, "crop": classifier.TIGHT_CROP, "top": [{"name": name, "score": score}],
            "local_share": local_share, **extra}


def check(score, empty=False):
    return {"model": "bioclip-presence", "score": score, "threshold": 0.02, "empty": empty}


def test_the_bars_sit_below_the_gates_and_come_from_config():
    assert pi_only.MODES == ("off", "shadow", "on") and config.PI_ONLY == "shadow"  # shadow first: it changes nothing
    assert (config.PI_ONLY_MIN_SCORE, config.PI_ONLY_HOLD_SCORE) == (0.75, 0.9)
    assert config.PI_ONLY_MIN_SCORE < config.GATE_MIN_SCORE_TIGHT  # else the gate would have named it already


def test_a_tight_look_between_the_two_bars_names_the_visit():
    v = pi_only.verdict(check(0.4), look("Dark-eyed Junco", 0.8), None, SETTLED, night=False)
    assert v == {"verdict": "named", "score": 0.4, "species": "Dark-eyed Junco", "visitor": "bird", "by": "tight"}
    # exactly the bar is enough; a squirrel whose belief leaked to squirrels abroad counts as local
    assert pi_only.verdict(check(0.4), look("Dark-eyed Junco", 0.75), None, SETTLED, False)["verdict"] == "named"
    leaked = look("Eastern Gray Squirrel", 0.86, local_share=0.02, kin_share=0.82)
    assert pi_only.verdict(check(0.1), leaked, None, SETTLED, False)["visitor"] == "squirrel"


@pytest.mark.parametrize("tight, night", [
    (look("Dark-eyed Junco", 0.74), False),                 # under the bar
    (look("Hermit Thrush", 0.83), False),                   # not settled here: what empty frames looked like
    (look("Dark-eyed Junco", 0.8, local_share=0.3), False),  # mostly on species from elsewhere, no relatives
    (look("Dark-eyed Junco", 0.99), True),                  # infrared: BioCLIP 2 never names at night
    (None, False),                                          # no finder found a patch, so no tight look
])
def test_a_look_that_may_not_name_falls_back_on_the_animal_checks_score(tight, night):
    assert pi_only.verdict(check(0.95), tight, None, SETTLED, night) == {"verdict": "held", "score": 0.95, "why": "score"}
    assert pi_only.verdict(check(0.9), tight, None, SETTLED, night)["verdict"] == "held"  # exactly the bar holds
    assert pi_only.verdict(check(0.89), tight, None, SETTLED, night) == {"verdict": "empty", "score": 0.89}


def test_no_verdict_from_the_animal_check_is_held_never_dropped():
    assert pi_only.verdict(None, None, None, SETTLED, False) == {"verdict": "held", "score": None, "why": "unchecked"}
    assert pi_only.verdict({"model": "bioclip-presence"}, None, None, SETTLED, True)["why"] == "unchecked"


def test_a_sure_first_look_stopped_only_by_the_crowd_names_the_visit():
    v = pi_only.verdict(check(1.0), None, "Eastern Gray Squirrel", SETTLED, False)
    assert v == {"verdict": "named", "score": 1.0, "species": "Eastern Gray Squirrel", "visitor": "squirrel", "by": "bowl"}
    assert pi_only.verdict(check(1.0), None, "Bushtit", SETTLED, False)["verdict"] == "held"  # not settled: no name


def test_gate_pick_takes_a_callers_bar_in_place_of_the_looks_own(monkeypatch):
    monkeypatch.setattr(classifier, "GATE_MIN_SCORE_TIGHT", 0.9)
    guess = look("Dark-eyed Junco", 0.8)
    assert classifier.gate_pick(guess, SETTLED, False, view="tight") is None
    assert classifier.gate_miss(guess, SETTLED, False, view="tight") == "score"
    assert classifier.gate_pick(guess, SETTLED, False, view="tight", bar=0.75) == ("Dark-eyed Junco", "bird")
    assert classifier.gate_miss(guess, SETTLED, False, view="tight", bar=0.85) == "score"
    assert classifier.gate_miss(guess, {}, False, view="tight", bar=0.75) == "unsettled"  # the other rules still hold


def test_a_held_frame_is_an_other_that_waits_for_the_owner():
    v = {"verdict": "held", "score": 0.934, "why": "score"}
    r = identify.settled_by_pi_only(v, check(0.934))
    assert (r.sighting.visitor, r.sighting.species, r.sighting.confidence) == ("other", None, "low")
    assert r.needs_review and not r.escalated and r.model == "bioclip-presence"  # "other" always waits (Result)
    assert r.raw == {"skipped_claude": True, "settled_by": "pi_only", "pi_only": v, "presence": check(0.934)}
    assert "Held for you" in r.sighting.notes and "0.934" in r.sighting.notes and "Claude is switched off (Settings > Pi checks)" in r.sighting.notes
    # no animal check at all: still held, and the note says so
    r = identify.settled_by_pi_only({"verdict": "held", "score": None, "why": "unchecked"}, None)
    assert r.needs_review and r.model == "bioclip-presence" and "presence" not in r.raw
    assert "gave no verdict" in r.sighting.notes


def test_an_empty_verdict_is_a_false_trigger_that_says_why():
    v = {"verdict": "empty", "score": 0.21}
    r = identify.settled_by_pi_only(v, check(0.21))
    assert (r.sighting.visitor, r.sighting.count, r.needs_review) == ("none", 0, False)
    assert r.raw["settled_by"] == "pi_only" and r.raw["skipped_claude"] and r.raw["pi_only"] == v
    assert "false trigger" in r.sighting.notes and "0.210" in r.sighting.notes and "0.90" in r.sighting.notes
