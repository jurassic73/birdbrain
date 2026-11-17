"""Unit tests of birdbrain/web/app and steps for the Pi-only verdict (birdbrain.pi_only): the visit page's sentence
and the decision trail's flags. Pure: JSON in, dicts out."""
import json

from birdbrain.web import app as webapp

NAMED = {"verdict": "named", "score": 0.4, "species": "Dark-eyed Junco", "visitor": "bird", "by": "tight"}
HELD = {"verdict": "held", "score": 0.934, "why": "score"}
EMPTY = {"verdict": "empty", "score": 0.21}


def test_the_shadow_sentence_says_what_the_pi_would_have_done_and_that_it_is_not_used():
    said = lambda record: webapp.pi_only_words({"pi_only": record, "haiku": {"answer": {"visitor": "bird"}}})
    assert said(NAMED) == ("Pi only (shadow): without Claude the Pi would have named it Dark-eyed Junco on the "
                           "tight look. Not used.", False)
    assert said({**NAMED, "by": "bowl"})[0] == ("Pi only (shadow): without Claude the Pi would have named it "
                                                "Dark-eyed Junco on the first look at the whole bowl. Not used.")
    assert said(HELD)[0] == ("Pi only (shadow): without Claude the Pi would have held this photo for you (the Pi "
                             "animal check scored it 0.93). Not used.")
    assert said(EMPTY)[0] == ("Pi only (shadow): without Claude the Pi would have filed this as a false trigger "
                              "(the Pi animal check scored it 0.21). Not used.")
    assert said({"verdict": "held", "score": None, "why": "unchecked"})[0].count("gave no verdict") == 1


def test_a_verdict_that_was_acted_on_says_claude_is_switched_off():
    on = lambda record: webapp.pi_only_words({"pi_only": record, "skipped_claude": True, "settled_by": "pi_only"})
    text, applied = on(HELD)
    assert applied and text.startswith("Claude is switched off (PI_ONLY=on), so the Pi decided alone: ")
    assert "the Pi animal check scored it 0.93" in text and "held for you" in text
    assert "filed as a false trigger" in on(EMPTY)[0] and on(EMPTY)[1]
    assert "the tight look named it Dark-eyed Junco" in on(NAMED)[0]


def test_no_record_or_a_broken_one_gives_no_sentence():
    assert webapp.pi_only_words({}) is None and webapp.pi_only_words({"pi_only": "named"}) is None
    assert webapp.pi_only_words({"pi_only": {"verdict": "maybe"}}) is None


def test_the_decision_trail_carries_the_sentence_and_whether_it_was_acted_on():
    shadow = webapp.decision_steps(json.dumps({"calls": {"pi_only": HELD, "haiku": {"answer": {"visitor": "none"}}}}))
    assert shadow["pi_only"].startswith("Pi only (shadow)") and shadow["pi_only_on"] is False
    on = webapp.decision_steps(json.dumps({"calls": {"pi_only": HELD, "skipped_claude": True, "settled_by": "pi_only"}}))
    assert on["pi_only"].startswith("Claude is switched off") and on["pi_only_on"] is True
    plain = webapp.decision_steps(json.dumps({"calls": {"haiku": {"answer": {"visitor": "none"}}}}))
    assert plain["pi_only"] is None and plain["pi_only_on"] is False
