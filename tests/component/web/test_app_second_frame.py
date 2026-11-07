"""Component tests of the web app through TestClient on a temp database: the shadow second-frame look on the Stats
page (the "Second-frame look (shadow)" paragraph) and the visit page (one sentence in the tight look step)."""

import re

from helpers.second_frame import second_frame_raw


def badges(html: str) -> int:
    return html.count('class="badge ')


def pills_explained(html: str) -> bool:
    return badges(html) == len(re.findall(r'class="badge [^"]*" data-tip=', html))


def test_stats_page_has_one_muted_sentence_without_any_second_frame_look(client, add_visit):
    add_visit(species="Bushtit", raw_json=second_frame_raw(("Dark-eyed Junco", 0.85), None, with_second=False))
    r = client.get("/stats")
    assert r.status_code == 200
    html = r.text
    section = html[html.index('id="classifier"'):]
    section = section[:section.index("</section>")]
    assert "second-frame look" in section.lower() or "second frame" in section.lower()
    assert "4 looks" not in section and "first frame right" not in section


def test_stats_page_scores_the_second_frame_look(client, add_visit):
    # 4 looks: first frame right 3, second frame right 2, averaged right 1; averaging would have named 2 (1 wrong)
    add_visit(species="Dark-eyed Junco", raw_json=second_frame_raw(("Dark-eyed Junco", 0.85), ("Dark-eyed Junco", 0.6),
                                                                   ("Bushtit", 0.97)))
    add_visit(species="Dark-eyed Junco", raw_json=second_frame_raw(("Dark-eyed Junco", 0.9), ("Bushtit", 0.5),
                                                                   ("Dark-eyed Junco", 0.96)))
    add_visit(species="Dark-eyed Junco", raw_json=second_frame_raw(("Dark-eyed Junco", 0.88), ("Bushtit", 0.9),
                                                                   ("Bushtit", 0.9)))
    add_visit(species="Dark-eyed Junco", raw_json=second_frame_raw(("Bushtit", 0.9), ("Dark-eyed Junco", 0.9)))
    html = client.get("/stats").text
    assert "Second-frame look (shadow)" in html
    seg = html[html.index("Second-frame look (shadow)"):]
    seg = seg[:seg.index("</section>")]
    assert re.search(r"\b4 looks\b|looks[^0-9]{0,20}\b4\b", seg)
    assert re.search(r"first frame[^0-9]{0,30}\b3\b", seg, re.I)
    assert re.search(r"second frame[^0-9]{0,30}\b2\b", seg, re.I)
    assert re.search(r"averaged[^0-9]{0,30}\b1\b", seg, re.I)
    assert re.search(r"named[^0-9]{0,30}\b2\b", seg, re.I)
    assert re.search(r"\b1 wrong\b|wrong[^0-9]{0,20}\b1\b", seg, re.I)
    assert pills_explained(html)


def test_visit_page_adds_one_sentence_for_the_second_frame_look(client, add_visit):
    plain = add_visit(species="Bushtit", raw_json=second_frame_raw(("Dark-eyed Junco", 0.85), None, with_second=False))
    looked = add_visit(species="Bushtit", raw_json=second_frame_raw(("Dark-eyed Junco", 0.85), ("Bushtit", 0.61),
                                                                    ("Dark-eyed Junco", 0.96)))
    before = client.get(f"/visit/{plain}").text
    html = client.get(f"/visit/{looked}").text
    sentence = "A second frame said Bushtit (0.61); averaged Dark-eyed Junco (0.96). Shadow only: not used."
    assert sentence in html
    steps = html[html.index("How it was decided"):]
    assert steps.index("Did not name the visit either: its top guess was 85%, under the 95% it needs.") < steps.index(sentence)
    assert badges(html) == badges(before) and pills_explained(html)  # no new pill


def test_visit_page_without_an_average_still_says_what_the_second_frame_said(client, add_visit):
    vid = add_visit(species="Bushtit", raw_json=second_frame_raw(("Dark-eyed Junco", 0.85), ("House Finch", 0.61)))
    html = client.get(f"/visit/{vid}").text
    assert "A second frame said House Finch (0.61)" in html and "Shadow only: not used." in html
    assert "averaged None" not in html and "(None)" not in html


def test_visits_without_a_second_frame_look_render_as_before(client, add_visit):
    vid = add_visit(species="Bushtit", raw_json=second_frame_raw(("Dark-eyed Junco", 0.85), None, with_second=False))
    html = client.get(f"/visit/{vid}").text
    assert "A second frame said" not in html and "Shadow only" not in html
    assert "Did not name the visit either: its top guess was 85%, under the 95% it needs. Claude was asked." in html
