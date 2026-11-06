"""The Lessons summary (birdbrain.lessons): Claude distills every note into points, drafts wait for the owner,
the accepted text goes into every prompt, a distill runs on its own every AUTO_EVERY notes, and admins get a pop-up."""

import json

import pytest

from helpers.lessons import fake_distill, notes, synchronous_distill  # noqa: F401  (autouse fixture)
from birdbrain import db, identify, lessons
from birdbrain.web import app as webapp


def test_prompt_site_carries_the_accepted_summary_to_every_caller(conn):
    assert db.prompt_site(conn)["lessons_summary"] == ""
    db.save_settings(conn, lessons_summary="- Rock tips are not birds.")
    assert db.prompt_site(conn)["lessons_summary"] == "- Rock tips are not birds."
    assert "<lessons_summary>" in identify.system_prompt(db.recent_lessons(conn), db.prompt_site(conn))


def test_distill_reads_every_lesson_oldest_first_except_switched_off_and_saves_a_draft(conn, add_visit, fake_distill):
    ids = notes(add_visit, 20)
    db.set_lesson_off(conn, ids[3], True)
    assert db.recent_lessons(conn) and len(db.recent_lessons(conn)) == db.LESSON_LIMIT
    result = lessons.distill(conn)
    assert result["ok"]
    lines = fake_distill[0]
    assert len(lines) == 19 and "number 0 " in lines[0] and "number 19 " in lines[-1]  # all of them, old ones too
    assert not any("number 3 " in l for l in lines)  # the owner switched that one off
    draft = db.get_json(conn, lessons.DRAFT)
    assert draft["text"] == "- Rock tips after a refill are not birds.\n- Juncos are the usual small bird."
    assert draft["notes"] == 19 and draft["auto"] is False and draft["changes"] == "Merged three rock notes."
    assert draft["model"] == lessons.MODEL and 0 < draft["usd"] < 0.01
    assert str(ids[0]) in draft["seen"] and str(ids[3]) not in draft["seen"]
    # a draft is never the summary
    assert db.prompt_site(conn)["lessons_summary"] == ""
    assert lessons.new_since(conn) == 0
    spend = db.get_json(conn, db.DISTILL_SPEND)
    assert spend["n"] == 1 and spend["usd"] == pytest.approx(draft["usd"], abs=1e-6)
    db.forget_totals()
    totals = db.totals(conn)
    assert totals["distills"] == 1 and totals["claude_usd"] >= totals["distill_usd"] > 0


def test_accept_puts_the_text_into_the_prompt_and_keeps_edits(conn, add_visit, fake_distill):
    notes(add_visit, 3)
    lessons.distill(conn)
    assert lessons.accept(conn, "Rock tips after a refill are not birds.\n\n- Only juncos so far\n") is True
    site = db.prompt_site(conn)
    assert site["lessons_summary"] == "- Rock tips after a refill are not birds.\n- Only juncos so far"
    assert "- Only juncos so far\n</lessons_summary>" in identify.system_prompt(db.recent_lessons(conn), site)
    meta = db.get_json(conn, lessons.SUMMARY_META)
    assert meta["edited"] is True and meta["notes"] == 3 and meta["accepted_at"]
    assert db.get_json(conn, lessons.DRAFT) is None
    assert lessons.accept(conn, "x") is False  # nothing left to accept


def test_accept_unedited_and_discard(conn, add_visit, fake_distill):
    notes(add_visit, 2)
    lessons.distill(conn)
    draft_text = db.get_json(conn, lessons.DRAFT)["text"]
    lessons.accept(conn)
    assert db.prompt_site(conn)["lessons_summary"] == draft_text
    assert db.get_json(conn, lessons.SUMMARY_META)["edited"] is False
    lessons.distill(conn)
    assert db.get_json(conn, lessons.DRAFT)
    lessons.discard(conn)
    assert db.get_json(conn, lessons.DRAFT) is None and db.prompt_site(conn)["lessons_summary"] == draft_text


def test_new_since_counts_notes_saved_or_changed_after_the_last_distill(conn, add_visit, fake_distill):
    ids = notes(add_visit, 4)
    assert lessons.new_since(conn) == 4  # never distilled: everything is new
    lessons.distill(conn)
    assert lessons.new_since(conn) == 0
    notes(add_visit, 2, start=10)
    db.correct_visit(conn, ids[0], "bird", "Bushtit", "it was a bushtit after all")  # changed: counts as new
    db.set_lesson_off(conn, ids[1], True)  # switched off: not counted either way
    assert lessons.new_since(conn) == 3
    status = lessons.status(conn)
    assert status["new"] == 3 and status["lessons"] == 5 and status["draft"]["notes"] == 4
    # which lessons the accepted summary covers, as they read now
    lessons.accept(conn)
    folded = lessons.in_summary(conn, db.all_lessons(conn))
    assert ids[2] in folded and ids[3] in folded and ids[0] not in folded and ids[1] not in folded


def test_distill_guards_against_a_longer_or_empty_answer_and_api_failures(conn, add_visit, monkeypatch):
    notes(add_visit, 2)
    long = lessons.Distilled(points=["many words " * 200], changes="")
    monkeypatch.setattr(lessons, "ask", lambda lines: (long, {"input_tokens": 1, "output_tokens": 1}))
    r = lessons.distill(conn)
    assert not r["ok"] and "wasn't shorter" in r["error"] and db.get_json(conn, lessons.DRAFT) is None
    monkeypatch.setattr(lessons, "ask", lambda lines: (lessons.Distilled(points=[" ", "-"], changes=""), {}))
    assert "empty" in lessons.distill(conn)["error"]
    monkeypatch.setattr(lessons, "ask", lambda lines: (None, {"input_tokens": 1, "output_tokens": 0}))
    assert "didn't give a summary" in lessons.distill(conn)["error"]

    def boom(lines):
        raise ConnectionError("down")
    monkeypatch.setattr(lessons, "ask", boom)
    r = lessons.distill(conn, auto=True)
    assert not r["ok"] and "ConnectionError" in r["error"]
    assert db.get_json(conn, lessons.FAILED)["error"] == r["error"]  # an automatic run remembers its failure
    assert lessons.distill(conn)["ok"] is False  # a manual run doesn't
    assert lessons.status(conn)["failed"]["error"] == r["error"]
    assert lessons.distill.__doc__  # and never raises


def test_distill_with_no_lessons_says_so(conn, fake_distill):
    r = lessons.distill(conn)
    assert not r["ok"] and "no lessons" in r["error"] and fake_distill == []


def test_automatic_distill_failure_is_recorded_and_retried_later(conn, add_visit, monkeypatch):
    notes(add_visit, lessons.AUTO_EVERY)
    calls = []

    def boom(lines):
        calls.append(1)
        raise TimeoutError("slow")
    monkeypatch.setattr(lessons, "ask", boom)
    assert lessons.maybe_auto(conn) is True  # tried
    assert db.get_json(conn, lessons.DRAFT) is None and db.get_json(conn, lessons.FAILED)
    assert lessons.maybe_auto(conn) is False and len(calls) == 1  # not again right away
    monkeypatch.setattr(lessons, "RETRY_AFTER", 0)
    monkeypatch.setattr(lessons, "_age", lambda at: 1.0)
    assert lessons.maybe_auto(conn) is True and len(calls) == 2
    assert lessons.maybe_auto.__doc__ and not lessons._running  # the flag is released after a failure


def test_automatic_distill_runs_once_at_a_time(conn, add_visit, fake_distill, monkeypatch):
    notes(add_visit, lessons.AUTO_EVERY)
    monkeypatch.setattr(lessons, "_running", True)
    assert lessons.maybe_auto(conn) is False and fake_distill == []
    monkeypatch.setattr(lessons, "_running", False)
    assert lessons.maybe_auto(conn) is True and len(fake_distill) == 1
    assert lessons.maybe_auto(conn) is False  # the draft waits for the owner


def test_popup_script_dismisses_per_browser_until_tomorrow():
    js = (webapp.HERE / "static" / "lessons-popup.js").read_text()
    assert "bb_lessons_later" in js and "24 * 3600 * 1000" in js and "l.at === at" in js
    assert 'querySelector("dialog[open]")' in js and "touring()" in js


def test_draft_json_round_trips(conn, add_visit, fake_distill):
    notes(add_visit, 1)
    lessons.distill(conn)
    raw = db.get_value(conn, lessons.DRAFT)
    assert json.loads(raw)["text"].startswith("- ")
