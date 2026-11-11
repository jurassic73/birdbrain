"""Component tests of the web app through TestClient on a temp database: the finder chain on the visit page, the
Stats page and About."""
import re

import pytest

from helpers.finder import finder_raw
from birdbrain import classifier

IN_WORDS = {"motion": "the Motion detector", "blob": "the crowd counter's outline",
            "yolox": "a small object detector (YOLOX-tiny)"}


def pills_explained(html):
    return html.count('class="badge ') == len(re.findall(r'class="badge [^"]*" data-tip=', html))


def tight_step(html):
    """The tight look's step on the visit page (its own <li> in the How it was decided list)."""
    steps = html[html.index("How it was decided"):]
    step = steps[steps.index('data-tip="Tight look:'):]
    return step[:step.index("</li>")]


@pytest.mark.parametrize("found", ["motion", "blob", "yolox"])
def test_visit_page_names_the_finder_of_a_tight_look_that_named_the_visit(client, add_visit, found):
    vid = add_visit(model=classifier.MODEL_NAME, raw_json=finder_raw(found, named=True))
    html = client.get(f"/visit/{vid}").text
    assert 'data-tip="Tight look:' in html and "Claude wasn't asked" in html
    step = tight_step(html).replace("&#39;", "'")
    assert IN_WORDS[found] in step
    for other, words in IN_WORDS.items():  # checked on the step only: an unrelated pill tip names the Motion detector
        if other != found:
            assert words not in step
    assert pills_explained(html)


@pytest.mark.parametrize("found", ["motion", "blob", "yolox"])
def test_visit_page_names_the_finder_of_a_tight_look_that_missed(client, add_visit, found):
    vid = add_visit(species="Bushtit", raw_json=finder_raw(found, score=0.9))
    html = client.get(f"/visit/{vid}").text
    assert "Did not name the visit either: its top guess was 90%, under the 95% it needs. Claude was asked." in html
    assert IN_WORDS[found] in tight_step(html).replace("&#39;", "'")
    assert pills_explained(html)


def test_visit_page_when_no_finder_found_a_patch(client, add_visit):
    vid = add_visit(species="Bushtit", raw_json=finder_raw(None))
    html = client.get(f"/visit/{vid}").text
    # no tight look: the first look's wording stays as it was
    assert "Did not name the visit: its top guess was 61%, under the 80% it needs. Claude was asked." in html
    assert 'data-tip="Tight look:' not in html and ">tight look</span>" not in html
    assert "Did not name the visit either" not in html
    # the tried list is a tip, not a paragraph: it names the finders that were tried
    tips = [t.replace("&#39;", "'") for t in re.findall(r'data-tip="([^"]*)"', html)]
    assert any("YOLOX" in t for t in tips), "no data-tip lists the finders tried"
    assert pills_explained(html)


def test_visits_from_before_the_finder_chain_render_as_before(client, add_visit):
    import json
    from helpers.classifier import MOTION_CIRCLE, MOTION_FOUND, guess
    old = add_visit(model=classifier.MODEL_NAME, raw_json=json.dumps({"calls": {
        "classifier": {**guess("Dark-eyed Junco"), "crop": classifier.TIGHT_CROP}, "skipped_claude": True,
        "settled_by": "classifier", "circle": MOTION_CIRCLE, "motion_circle": MOTION_FOUND}}))
    html = client.get(f"/visit/{old}").text
    assert 'data-tip="Tight look:' in html and "Claude wasn't asked" in html and pills_explained(html)
    page = html.replace("&#39;", "'")
    assert IN_WORDS["blob"] not in page and IN_WORDS["yolox"] not in page  # nothing was tried but motion back then


def test_stats_page_renders_the_finder_line_with_and_without_data(client, add_visit):
    response = client.get("/stats")
    assert response.status_code == 200 and 'id="classifier"' in response.text  # the section renders with no looks
    for _ in range(5):
        add_visit(model=classifier.MODEL_NAME, raw_json=finder_raw("blob", named=True))
    for _ in range(2):
        add_visit(species="Bushtit", raw_json=finder_raw("motion"))
    for _ in range(3):
        add_visit(species="Bushtit", raw_json=finder_raw(None))
    response = client.get("/stats")
    assert response.status_code == 200
    section = response.text[response.text.index('id="classifier"'):]
    section = section[:section.index("</section>")].replace("&#39;", "'")
    # 7 tight looks, 3 with no patch; blob: 5 looks, 5 named; motion: 2 looks, none named
    assert re.search(r"crowd counter|blob", section, re.I) and re.search(r"motion", section, re.I)
    for n in ("7", "5", "3", "2"):
        assert n in section


def test_about_page_mentions_the_other_finders_in_a_sentence(client):
    html = client.get("/about").text.replace("&#39;", "'")
    assert "crowd counter" in html and "object detector" in html
    # The rain rule looks at the photos, not the weather: said in a sentence, with the Pi animal check
    assert "Rain on the water:</b> judged from the photos themselves, never from a weather forecast." in html
    assert "still being tuned" in html
    # YOLOX's other job (keeping people off the public mirror) stays off the About page, the owner's choice
    assert not re.search(r"YOLOX[^.<]*privacy|privacy[^.<]*YOLOX", html, re.I)
