"""Component tests of the web app through TestClient on a temp database: also."""

import json
import re

from helpers.also import BUSHTITS
from birdbrain import db
from birdbrain.web import app as webapp


def test_owner_list_wins_and_undo_brings_back_claudes(client, conn, add_visit):
    vid = add_visit(also=json.dumps(BUSHTITS))
    client.post(f"/visit/{vid}/correct", data={"visitor": "bird", "species": "Dark-eyed Junco",
                                               "also_species": ["Spotted Towhee"], "also_count": ["1"]})
    v = db.get_visit(conn, vid)
    assert [db.also_label(a) for a in db.also_of(v)] == ["Spotted Towhee"]
    assert db.count_visits(conn, species="Bushtit") == 0 and db.count_visits(conn, species="Spotted Towhee") == 1
    line = db.recent_lessons(conn)[0]
    assert 'you said "Dark-eyed Junco" with "2 Bushtit"' in line
    assert 'The owner says also there: "1 Spotted Towhee".' in line
    page = client.get(f"/visit/{vid}").text
    assert "Corrected by you. Claude said: Dark-eyed Junco with Bushtit x2." in page
    client.post(f"/visit/{vid}/clear-correction")
    assert db.count_visits(conn, species="Bushtit") == 1 and db.recent_lessons(conn) == []


def test_removing_every_other_animal_is_a_lesson(client, conn, add_visit):
    vid = add_visit(also=json.dumps(BUSHTITS))
    client.post(f"/visit/{vid}/correct", data={"visitor": "bird", "species": "Dark-eyed Junco"})
    assert db.recent_lessons(conn)[0].endswith("The owner says no other animal was there.")
    assert db.totals(conn)["corrected"] == 1


def test_editing_a_lesson_keeps_the_owners_other_animals(client, conn, add_visit):
    vid = add_visit(also=json.dumps(BUSHTITS))
    client.post(f"/visit/{vid}/correct", data={"also_species": ["Bushtit"], "also_count": ["5"]})
    client.post(f"/lessons/{vid}/edit", data={"visitor": "bird", "species": "Song Sparrow", "feedback": "x"})
    assert db.also_of(db.get_visit(conn, vid))[0]["count"] == 5


def test_cards_and_visit_page_show_the_other_animals(client, conn, add_visit):
    circle = {"x": 0.45, "y": 0.65, "r": 0.05, "claude": True}
    vid = add_visit(also=json.dumps(BUSHTITS), raw_json=json.dumps({"calls": {"circle": circle}}))
    assert '<span class="also-name">+ <span class="sp-name" data-species="Bushtit"' in client.get("/").text
    page = client.get(f"/visit/{vid}").text
    assert '<p class="also-line">With <span class="sp-name" data-species="Bushtit"' in page
    assert 'name="also_species"' in page and 'value="Bushtit"' in page
    marks = json.loads(page.split('data-marks="')[1].split('"')[0].replace("&#34;", '"'))
    assert [m.get("label") for m in marks] == ["Dark-eyed Junco", "Bushtit"]
    client.post(f"/visit/{vid}/correct", data={"feedback": "ok", "marks": json.dumps(marks),
                                               "also_species": ["Bushtit"], "also_count": ["2"]})
    assert "circle 2 (you drew it around the Bushtit)" in db.recent_lessons(conn)[0]
    assert "Bushtit" in client.get("/species/Bushtit").text


def test_one_circle_needs_no_label(client, add_visit):
    vid = add_visit(raw_json=json.dumps({"calls": {"circle": {"x": 0.4, "y": 0.4, "r": 0.05, "claude": True}}}))
    assert webapp.start_marks(db.get_visit(webapp.db.connect(), vid)) == [
        {"x": 0.4, "y": 0.4, "r": 0.05, "claude": True}]


def test_visit_page_lists_other_animals_in_an_also_row(client, add_visit):
    html = client.get(f"/visit/{add_visit(also=json.dumps(BUSHTITS))}").text
    assert "<dt>Also</dt><dd>Bushtit x2 <span" in html and "<dt>Species</dt>" not in html
    assert "<dt>Also</dt>" not in client.get(f"/visit/{add_visit()}").text


def test_each_circle_is_saved_with_the_animal_it_holds(client, conn, add_visit):
    vid = add_visit(also=json.dumps(BUSHTITS))
    page = client.get(f"/visit/{vid}").text
    # a slot for the circles' numbers in front of the Species box and of every "Also in the photo" row
    assert re.search(r'<div class="ask-wrap">\s*(\{#.*?#\}\s*)?<span class="circle-nums" hidden></span>\s*<div class="combo">\s*<input id="species"', page)
    assert page.count('<li class="also-row">\n        <span class="circle-nums" hidden></span>') == 2  # the row and the template
    circles = [{"x": 0.3, "y": 0.3, "r": 0.05, "label": "bushtit"}, {"x": 0.6, "y": 0.6, "r": 0.05, "label": "Dark-eyed Junco"},
               {"x": 0.8, "y": 0.2, "r": 0.05, "label": "Steller's Jay"}]
    client.post(f"/visit/{vid}/correct", data={"feedback": "two kinds", "marks": json.dumps(circles),
                                               "also_species": ["Bushtit"], "also_count": ["2"]})
    saved = db.parse_marks(db.get_visit(conn, vid)["marks"])
    assert [m.get("label") for m in saved] == ["Bushtit", "Dark-eyed Junco", None]  # the jay isn't in this visit
    lesson = db.recent_lessons(conn)[0]
    assert "circle 1 (the owner drew it around the Bushtit)" in lesson and "circle 3 at 80% across" in lesson
    # a new main species: its circle is renamed by the form before saving; a stale name is not kept
    client.post(f"/visit/{vid}/correct", data={"visitor": "bird", "species": "Song Sparrow", "marks": json.dumps(saved),
                                               "also_species": ["Bushtit"], "also_count": ["2"]})
    assert [m.get("label") for m in db.parse_marks(db.get_visit(conn, vid)["marks"])] == ["Bushtit", None, None]
    # the names reach the page for everyone who looks at the photo
    assert "&#34;label&#34;: &#34;Bushtit&#34;" in client.get(f"/visit/{vid}").text
