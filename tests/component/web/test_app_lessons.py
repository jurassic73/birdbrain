"""Component tests of the web app through TestClient on a temp database: lessons."""

import re

from helpers.lessons import fake_distill, notes, synchronous_distill  # noqa: F401  (autouse fixture)
from birdbrain import db, lessons


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
    assert client.post("/notes/summary/distill").status_code == 404  # no admin on the mirror: as if it didn't exist


def test_summary_routes_are_admin_only(client, conn, add_visit, locked):
    notes(add_visit, 1)
    for path in ("/notes/summary/distill", "/notes/summary/accept", "/notes/summary/discard"):
        assert client.post(path).status_code == 403
    client.post("/login", data={"password": "s3cret!"})
    assert client.post("/notes/summary/discard", follow_redirects=False).status_code == 303


def test_about_page_mentions_the_summary(client, conn):
    html = client.get("/about").text
    assert "Lessons summary" in html and "in-context learning, not fine-tuning" in html
    assert f"after every {lessons.AUTO_EVERY} new notes" in html and "none accepted yet" in html
    db.save_settings(conn, lessons_summary="- Rock tips are not birds.")
    assert "none accepted yet" not in client.get("/about").text
