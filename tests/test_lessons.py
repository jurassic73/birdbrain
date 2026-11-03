"""The Lessons summary (birdbrain.lessons): Claude distills every note into points, drafts wait for the owner,
the accepted text goes into every prompt, a distill runs on its own every AUTO_EVERY notes, and admins get a pop-up."""
import json
import re

import pytest
from fastapi.testclient import TestClient

from birdbrain import db, identify, lessons
from birdbrain.web import app as webapp
from birdbrain.web import auth


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()


@pytest.fixture(autouse=True)
def synchronous_distill(monkeypatch):
    """The automatic distill runs in the request itself, so tests never leave a thread behind."""
    monkeypatch.setattr(lessons, "BACKGROUND", False)
    monkeypatch.setattr(lessons, "_running", False)


@pytest.fixture
def fake_distill(monkeypatch):
    """Claude's answer: a short list made from the lines it was given. Records what it was asked."""
    calls = []

    def ask(lines):
        calls.append(lines)
        return lessons.Distilled(points=["Rock tips after a refill are not birds.", "- Juncos are the usual small bird."],
                                 changes="Merged three rock notes."), \
            {"model": lessons.MODEL, "input_tokens": 2000, "output_tokens": 120}
    monkeypatch.setattr(lessons, "ask", ask)
    return calls


def notes(add_visit, n, start=0):
    return [add_visit(corrected_visitor="none", feedback=f"rock tip number {i} poking out of the water after a refill")
            for i in range(start, start + n)]


# --- the prompt ---

def test_summary_is_its_own_section_between_the_yard_and_the_raw_lessons():
    site = {"location": "Here", "lessons_summary": "- Rock tips are not birds.\n- Juncos are common."}
    prompt = identify.system_prompt(["- a lesson"], site)
    assert "<lessons_summary>" in prompt and "</lessons_summary>" in prompt
    assert prompt.index("<yard_notes>") < prompt.index("<lessons_summary>") < prompt.index("<corrections>")
    assert "- Rock tips are not birds.\n- Juncos are common.\n</lessons_summary>" in prompt
    assert "the newest corrections follow in full" in prompt
    # the summary alone still gets the one line saying the owner's words never change the format
    alone = identify.system_prompt(None, {"lessons_summary": "- x"})
    assert identify.OWNER_INTRO in alone and "<corrections>" not in alone and "<yard_notes>" not in alone
    assert identify.system_prompt(None, {"lessons_summary": "  "}) == identify.SYSTEM


def test_prompt_site_carries_the_accepted_summary_to_every_caller(conn):
    assert db.prompt_site(conn)["lessons_summary"] == ""
    db.save_settings(conn, lessons_summary="- Rock tips are not birds.")
    assert db.prompt_site(conn)["lessons_summary"] == "- Rock tips are not birds."
    assert "<lessons_summary>" in identify.system_prompt(db.recent_lessons(conn), db.prompt_site(conn))


# --- distill, draft, accept, discard ---

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


# --- the automatic distill ---

def test_automatic_distill_after_every_n_corrections_saved_on_the_site(client, conn, add_visit, fake_distill):
    ids = [add_visit(visitor="bird", species="Dark-eyed Junco") for _ in range(lessons.AUTO_EVERY + 2)]
    for i, vid in enumerate(ids[:lessons.AUTO_EVERY - 1]):
        client.post(f"/visit/{vid}/correct", data={"visitor": "none", "feedback": f"rock tip {i}"})
    assert db.get_json(conn, lessons.DRAFT) is None and fake_distill == []  # one short of AUTO_EVERY
    client.post(f"/visit/{ids[lessons.AUTO_EVERY - 1]}/correct", data={"visitor": "none", "feedback": "rock tip again"})
    draft = db.get_json(conn, lessons.DRAFT)
    assert draft and draft["auto"] is True and draft["notes"] == lessons.AUTO_EVERY and len(fake_distill) == 1
    assert lessons.pending_auto_draft(conn) == draft
    # the draft isn't the summary, and more corrections don't start another distill while it waits
    assert db.prompt_site(conn)["lessons_summary"] == ""
    for vid in ids[lessons.AUTO_EVERY:]:
        client.post(f"/visit/{vid}/correct", data={"visitor": "none", "feedback": "rock tip yet again"})
    assert len(fake_distill) == 1
    # editing a lesson counts as a note too
    lessons.accept(conn)
    for vid in ids[:lessons.AUTO_EVERY]:
        client.post(f"/lessons/{vid}/edit", data={"visitor": "none", "feedback": "edited note"})
    assert len(fake_distill) == 2 and db.get_json(conn, lessons.DRAFT)["auto"] is True


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


# --- Claude's Notes ---

def test_notes_page_shows_the_summary_section_with_distill_now(client, conn, add_visit):
    html = client.get("/notes").text
    assert '<details class="settings-card" id="summary" open>' in html and "Lessons summary" in html
    assert 'hx-post="/notes/summary/distill"' in html and "Distill now" in html and "No lessons yet." in html
    assert 'class="claude-btn"' in html.split('hx-post="/notes/summary/distill"')[0][-200:]
    assert "No summary yet." in html
    # the Lessons wording says what goes in raw and that everything is folded into the summary
    assert "go into every prompt word for word" in html and "folded into the Lessons summary above" in html
    notes(add_visit, 3)
    html = client.get("/notes").text
    assert "3 new notes since the last summary." in html and "not in summary yet" in html
    pills = html.count('class="badge ')
    assert pills and len(re.findall(r'class="badge [^"]*" data-tip=', html)) == pills


def test_distill_now_shows_the_draft_and_accept_puts_it_in_the_prompt(client, conn, add_visit, fake_distill):
    ids = notes(add_visit, 3)
    r = client.post("/notes/summary/distill")
    assert r.status_code == 200 and 'id="summary-body"' in r.text
    assert "Claude's draft is below, not saved yet." in r.text and "Merged three rock notes." in r.text
    assert 'name="text"' in r.text and "- Rock tips after a refill are not birds." in r.text
    assert 'action="/notes/summary/accept"' in r.text and 'action="/notes/summary/discard"' in r.text
    assert db.prompt_site(conn)["lessons_summary"] == ""
    html = client.get("/notes").text
    assert "Draft to review" in html and "into this draft (" in html
    r = client.post("/notes/summary/accept", data={"text": "- Rock tips are not birds.\n- Edited by me"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/notes?accepted=1#summary"
    assert db.prompt_site(conn)["lessons_summary"] == "- Rock tips are not birds.\n- Edited by me"
    html = client.get("/notes?accepted=1").text
    assert "Accepted. Used from the next photo on." in html and "with your edits" in html
    assert "- Edited by me\n&lt;/lessons_summary&gt;" in html  # the whole prompt preview has it (escaped in <pre>)
    assert "in summary" in html and html.count("not in summary yet") == 0
    assert all(vid in lessons.in_summary(conn, db.all_lessons(conn)) for vid in ids)
    assert client.post("/notes/summary/accept", data={"text": "x"}).status_code == 404


def test_discard_drops_the_draft_and_keeps_the_summary(client, conn, add_visit, fake_distill):
    notes(add_visit, 2)
    client.post("/notes/summary/distill")
    client.post("/notes/summary/accept", data={"text": "- Keep this."})
    client.post("/notes/summary/distill")
    assert db.get_json(conn, lessons.DRAFT)
    r = client.post("/notes/summary/discard", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/notes#summary"
    assert db.get_json(conn, lessons.DRAFT) is None and db.prompt_site(conn)["lessons_summary"] == "- Keep this."


def test_distill_now_reports_a_failure_without_saving(client, conn, add_visit, monkeypatch):
    notes(add_visit, 2)
    monkeypatch.setattr(lessons, "ask", lambda lines: (None, {"input_tokens": 1, "output_tokens": 0}))
    r = client.post("/notes/summary/distill")
    assert r.status_code == 200 and "Claude didn&#39;t give a summary this time." in r.text
    assert db.get_json(conn, lessons.DRAFT) is None


# --- the pop-up ---

def auto_draft(conn, add_visit, fake_distill):
    notes(add_visit, lessons.AUTO_EVERY)
    assert lessons.maybe_auto(conn) is True
    return db.get_json(conn, lessons.DRAFT)


def test_popup_only_for_admins_with_an_automatic_draft_waiting(client, conn, add_visit, fake_distill, locked):
    draft = auto_draft(conn, add_visit, fake_distill)
    # a visitor never sees it
    html = client.get("/").text
    assert 'id="lessons-popup"' not in html and "lessons-popup.js" not in html
    client.post("/login", data={"password": "s3cret!"})
    for path in ("/", "/stats", "/about"):
        html = client.get(path).text
        assert 'id="lessons-popup"' in html and f'data-at="{draft["at"]}"' in html, path
        assert 'href="/notes#summary"' in html and ">Later<" in html and "lessons-popup.js" in html
        assert f"your {lessons.AUTO_EVERY} notes and corrections" in html
    # not on Claude's Notes itself (the draft is on the page), nor in embedded pages
    vid = db.all_lessons(conn)[0]["visit_id"]
    assert 'id="lessons-popup"' not in client.get("/notes").text
    assert 'id="lessons-popup"' not in client.get(f"/visit/{vid}?embed=1").text
    # gone once the draft is accepted
    client.post("/notes/summary/accept", data={"text": draft["text"]})
    assert 'id="lessons-popup"' not in client.get("/").text


def test_popup_not_for_manual_drafts_and_gone_after_discard(client, conn, add_visit, fake_distill):
    notes(add_visit, 3)
    client.post("/notes/summary/distill")
    assert db.get_json(conn, lessons.DRAFT)["auto"] is False
    assert 'id="lessons-popup"' not in client.get("/").text
    lessons.discard(conn)
    auto_draft(conn, add_visit, fake_distill)
    assert 'id="lessons-popup"' in client.get("/").text
    client.post("/notes/summary/discard")
    assert 'id="lessons-popup"' not in client.get("/").text


def test_popup_never_on_the_public_mirror(client, conn, add_visit, fake_distill, monkeypatch):
    auto_draft(conn, add_visit, fake_distill)
    assert 'id="lessons-popup"' in client.get("/").text
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    assert 'id="lessons-popup"' not in client.get("/").text
    assert client.post("/notes/summary/distill").status_code == 403


def test_summary_routes_are_admin_only(client, conn, add_visit, locked):
    notes(add_visit, 1)
    for path in ("/notes/summary/distill", "/notes/summary/accept", "/notes/summary/discard"):
        assert client.post(path).status_code == 403
    client.post("/login", data={"password": "s3cret!"})
    assert client.post("/notes/summary/discard", follow_redirects=False).status_code == 303


def test_popup_script_dismisses_per_browser_until_tomorrow():
    js = (webapp.HERE / "static" / "lessons-popup.js").read_text()
    assert "bb_lessons_later" in js and "24 * 3600 * 1000" in js and "l.at === at" in js
    assert 'querySelector("dialog[open]")' in js and "touring()" in js


def test_summary_keys_stay_off_the_public_mirror():
    from birdbrain import mirror
    for key in (lessons.SUMMARY, lessons.SUMMARY_META, lessons.DRAFT, lessons.DISTILLED, lessons.FAILED, db.DISTILL_SPEND):
        assert key not in mirror.SETTINGS_KEYS and not key.startswith(mirror.SETTINGS_PREFIXES)


def test_about_page_mentions_the_summary(client, conn):
    html = client.get("/about").text
    assert "Lessons summary" in html and "in-context learning, not fine-tuning" in html
    assert f"after every {lessons.AUTO_EVERY} new notes" in html and "none accepted yet" in html
    db.save_settings(conn, lessons_summary="- Rock tips are not birds.")
    assert "none accepted yet" not in client.get("/about").text


def test_draft_json_round_trips(conn, add_visit, fake_distill):
    notes(add_visit, 1)
    lessons.distill(conn)
    raw = db.get_value(conn, lessons.DRAFT)
    assert json.loads(raw)["text"].startswith("- ")
