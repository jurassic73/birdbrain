"""Component tests of the web app through TestClient on a temp database: presence."""

import json

from fastapi.testclient import TestClient

from helpers.presence import INSIDE, FakeClf, burst_visit, found_animal, on_frame, vec, yard
from birdbrain import db, presence
from birdbrain.web import app as webapp


def test_looks_right_confirms_and_undo_takes_it_back(client, conn, add_visit):
    vid = add_visit()
    assert ">Confirm</button>" in client.get(f"/visit/{vid}").text
    client.post(f"/visit/{vid}/reviewed")
    assert db.get_visit(conn, vid)["confirmed_at"]
    page = client.get(f"/visit/{vid}").text
    assert "confirmed by human" in page and ">Confirm</button>" not in page
    client.post(f"/visit/{vid}/unconfirm")
    assert db.get_visit(conn, vid)["confirmed_at"] is None


def test_confirm_says_no_animal_only_on_empty_frames(client, add_visit):
    assert "No animal in image:" not in client.get(f"/visit/{add_visit()}").text
    empty = add_visit(visitor="none", species=None, count=0)
    assert "No animal in image:</span><button" in client.get(f"/visit/{empty}").text


def test_false_triggers_confirm_all_shown_skips_real_visits(client, conn, add_visit):
    empty = [add_visit(visitor="none", species=None) for _ in range(3)]
    real = add_visit()
    page = client.get("/false-triggers").text
    assert ">3</span>" in page and "Confirm all shown false triggers have no animals</button>" in page
    assert "not contain an animal" in page  # the pop-up before it confirms
    assert 'href="#ft-storage"' in page and 'id="ft-storage"' in page  # the intro links to the Delete button
    assert "Motion, but no animal found. Open each photo to check it." in page and 'id="ft-visit"' in page
    r = client.post("/false-triggers/confirm", data={"ids": ",".join(map(str, [*empty[:2], real]))},
                    follow_redirects=False)
    assert r.headers["location"] == "/false-triggers?confirmed=2"
    assert [bool(db.get_visit(conn, i)["confirmed_at"]) for i in [*empty, real]] == [True, True, False, False]
    assert db.count_unchecked_false_triggers(conn) == 1


def test_false_triggers_list_only_unconfirmed_unless_show_all(client, conn, add_visit):
    done, todo = add_visit(visitor="none", species=None), add_visit(visitor="none", species=None)
    db.confirm_visits(conn, [done])
    page = client.get("/false-triggers").text
    assert f'data-id="{todo}"' in page and f'data-id="{done}"' not in page
    assert "not yet confirmed" not in page and "Show confirmed (1)" in page
    page = client.get("/false-triggers?show=all").text
    assert f'data-id="{todo}"' in page and f'data-id="{done}"' in page and "Hide confirmed" in page
    assert "data-show-all" in page and "have no animals</button>" not in page  # bulk confirm only on the to-check list
    db.confirm_visits(conn, [todo])
    assert "Nothing left to check" in client.get("/false-triggers").text


def test_settings_switches_the_animal_check(client, conn):
    assert "Pi checks" in client.get("/settings").text
    client.post("/settings/presence", data={"mode": "gate"})
    assert presence.mode(conn) == "gate"
    assert client.post("/settings/presence", data={"mode": "nope"}).status_code == 400


def test_the_visit_page_says_which_frame_teaches_the_check(conn, yard, add_visit):
    vid = burst_visit(conn, add_visit, [60, 200])
    found_animal(conn, vid, circle=on_frame(INSIDE, 2))
    presence.pick_frames(conn, clf=FakeClf(yard))
    with TestClient(webapp.app) as c:
        html = c.get(f"/visit/{vid}").text
    assert "burst frame 2 (your circle)" in html and "An animal example" in html
    assert 'class="save-btn"' in html and "your first circle is on another burst frame" in html


def test_the_visit_page_shows_both_frames_scores(client, conn, add_visit):
    vid = add_visit(raw_json=json.dumps({"calls": {"presence": {
        "model": presence.MODEL_NAME, "score": 0.8, "threshold": 0.1, "empty": False, "trained_on": 700,
        "frames": {"sharpest": 0.01, "last": 0.8}, "frame": "last"}}}))
    page = client.get(f"/visit/{vid}").text
    assert "the burst's last frame 0.800" in page and "so the last frame became the photo" in page


def test_the_note_box_says_claude_wasnt_asked_on_a_skipped_visit(client, add_visit):
    skipped = add_visit(visitor="none", species=None, count=0, model=presence.MODEL_NAME, raw_json=json.dumps(
        {"calls": {"skipped_claude": True, "presence": {"score": 0.01, "threshold": 0.1, "empty": True, "trained_on": 700}}}))
    page = client.get(f"/visit/{skipped}").text
    assert "Claude didn't look at this one. Describe each numbered circle, e.g. 1: spiderweb moving in the wind" in page
    assert "Claude wasn't asked about this visit" in page
    asked = client.get(f"/visit/{add_visit(visitor='none', species=None, count=0)}").text
    assert 'placeholder="Describe each numbered circle, e.g. 1: spiderweb moving in the wind"' in asked
    assert "Claude wasn't asked about this visit" not in asked


def trains_box(client, vid) -> str:
    html = client.get(f"/visit/{vid}").text
    start = html.index('id="trains"')
    return html[start:html.index("</section>", start)]


def pi_settled_empty(add_visit, conn, yard):
    vid = add_visit(visitor="none", species=None, count=0, model=presence.MODEL_NAME, raw_json=json.dumps(
        {"calls": {"skipped_claude": True, "presence": {"score": 0.01, "threshold": 0.1, "empty": True, "trained_on": 700}},
         "trigger": "rim"}))
    presence.store(conn, vid, vec(False, yard))
    return vid


def test_the_box_for_an_empty_photo_the_pi_settled_and_nobody_checked(client, conn, yard, add_visit):
    box = trains_box(client, pi_settled_empty(add_visit, conn, yard))
    for name in ("Motion detector", "Pi animal check", "BioCLIP 2 species check", "Claude"):
        assert name in box
    assert "An empty frame." in box and "Counts 3x once you Confirm" in box and "counts 3x on Confirm" in box
    assert "Not reached" in box and "not reached" in box
    assert "Not asked" in box and "no lesson yet" in box and "Confirm alone is not a lesson" in box
    assert "It spotted movement on the rim" in box


def test_the_box_after_confirming_says_counted_3x(client, conn, yard, add_visit):
    vid = pi_settled_empty(add_visit, conn, yard)
    db.confirm_visits(conn, [vid])
    box = trains_box(client, vid)
    assert "An empty frame, counted 3x" in box and ">counted 3x<" in box
    assert "no lesson yet" in box  # Confirm alone is not a lesson


def test_the_box_for_a_false_trigger_corrected_to_an_animal(client, conn, yard, add_visit):
    vid = pi_settled_empty(add_visit, conn, yard)
    found_animal(conn, vid)
    box = trains_box(client, vid)
    assert "An animal example" in box and "counted 3x: you checked it" in box
    assert "An empty frame" not in box
    assert "Not reached" in box and "Not asked" in box
    assert ">lesson<" in box and "A lesson on Claude" in box  # a correction is a lesson, sent word for word


def test_the_box_for_a_visit_claude_answered(client, conn, yard, add_visit):
    vid = add_visit(raw_json=json.dumps({"calls": {"haiku": {"frames": 1, "answer": {
        "visitor": "bird", "species": "Dark-eyed Junco", "confidence": "high"}}}}))
    presence.store(conn, vid, vec(True, yard))
    box = trains_box(client, vid)
    assert "Answered:" in box and "Dark-eyed Junco" in box and "An animal example" in box
    assert "No guess" in box  # no BioCLIP 2 species check step was recorded


def test_the_visit_box_is_for_admins_only(client, conn, yard, add_visit, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    assert 'id="trains"' not in client.get(f"/visit/{pi_settled_empty(add_visit, conn, yard)}").text


def test_the_box_for_a_pi_named_visit_the_owner_changed_or_confirmed(client, conn, yard, add_visit):
    """A visit the BioCLIP 2 species check named counts toward that species being settled only once the owner
    confirms it or names a species (db.settled_species); "No animal" or a bare type does not count."""
    raw = {"calls": {"skipped_claude": True, "settled_by": "classifier", "classifier": {
        "top": [{"name": "Dark-eyed Junco", "score": 0.97, "scientific": "Junco hyemalis"}], "local_share": 0.9}}}
    vid = add_visit(model="bioclip-2", species="Dark-eyed Junco", raw_json=json.dumps(raw))
    presence.store(conn, vid, vec(True, yard))
    box = trains_box(client, vid)
    assert "Named the animal on the Pi" in box and "Press Looks right" in box
    db.correct_visit(conn, vid, "none", None, None)  # the owner: nothing was there
    box = trains_box(client, vid)
    assert "counts toward Dark-eyed Junco being settled" not in box and "does not count" in box
    db.correct_visit(conn, vid, "squirrel", None, None)  # a type with no name counts toward nothing either
    assert "does not count" in trains_box(client, vid)
    db.correct_visit(conn, vid, None, "Bushtit", None)
    assert "counts toward Bushtit being settled" in trains_box(client, vid)
    db.correct_visit(conn, vid, None, None, None)
    db.confirm_visits(conn, [vid])
    assert "counts toward Dark-eyed Junco being settled" in trains_box(client, vid)


def test_the_visit_page_says_why_a_frame_was_taken_for_rain(client, add_visit):
    rain = add_visit(visitor="none", species=None, count=0, model=presence.MODEL_NAME, raw_json=json.dumps(
        {"trigger": "rim", "calls": {"skipped_claude": True, "settled_by": "rain", "rain": {"score_max": 0.3},
                                     "presence": {"score": 0.041, "threshold": 0.02, "empty": False, "trained_on": 627},
                                     "classifier_gate": {"model": "bioclip-2", "crop": "bowl", "night": False,
                                                         "seconds": 0.0, "why": "score", "local_share": 0.9,
                                                         "top": [{"name": "Varied Thrush", "scientific": "", "score": 0.41}]}}}))
    page = client.get(f"/visit/{rain}").text
    assert "taken for rain on the water (step 1), so Claude wasn't asked" in page and "Claude was asked." not in page
    assert "Probably rain on the water" in page and "(score 0.041, empty below 0.020)" in page
    assert "Not judged empty, but the score is low (under 0.30) and nothing new showed on land" in page
    assert "Judged empty, so Claude wasn't asked" not in page
    assert "Claude wasn't asked about this visit" in page  # the note box, as for any frame the Pi kept


def test_settings_counts_the_frames_taken_for_rain(client, add_visit):
    assert "taken for rain" not in client.get("/settings").text  # nothing to show yet
    add_visit(visitor="none", species=None, count=0, model=presence.MODEL_NAME, raw_json=json.dumps(
        {"calls": {"skipped_claude": True, "settled_by": "rain",
                   "presence": {"score": 0.041, "threshold": 0.02, "empty": False}}}))
    page = client.get("/settings").text
    assert '<span class="big">1</span> taken for rain' in page
    assert "daytime rim triggers it takes for rain on the water" in page
