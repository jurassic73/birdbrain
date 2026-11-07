"""Component tests of the web app through TestClient on a temp database: the finder wording carried over from the
finder chain (the first-look step, the training card, the two pill tips and the circle step's "pointed out by"
sentence) and the About page's sentence. The second-frame look itself is tested in test_app_second_frame.py."""
import json
import re

import pytest

from helpers.classifier import MOTION_CIRCLE, guess
from helpers.finder import finder_raw
from birdbrain import classifier

AT = {"motion": "at the patch that moved", "blob": "at the crowd counter's outline",
      "yolox": "at the box a small object detector (YOLOX-tiny) drew"}


def pills_explained(html):
    return html.count('class="badge ') == len(re.findall(r'class="badge [^"]*" data-tip=', html))


@pytest.mark.parametrize("found", ["motion", "blob", "yolox"])
def test_first_look_step_and_training_card_say_where_the_tight_look_looked(client, add_visit, found):
    html = client.get(f"/visit/{add_visit(species='Bushtit', raw_json=finder_raw(found, score=0.9))}").text
    page = html.replace("&#39;", "'")
    assert f"The Pi took a tight look {AT[found]} next." in page
    assert f"Then a tight look {AT[found]} guessed Dark-eyed Junco, but its top guess was 90%" in page
    for other, words in AT.items():
        if other != found:
            assert f"tight look {words}" not in page
    html = client.get(f"/visit/{add_visit(model=classifier.MODEL_NAME, raw_json=finder_raw(found, named=True))}").text
    page = html.replace("&#39;", "'")
    assert f"on a tight look {AT[found]} after a first look at the whole bowl fell short." in page
    assert f"The Pi took a tight look {AT[found]} next." in page


def test_the_pill_tips_hold_for_every_finder_and_the_step_is_named_after_the_pi(client, add_visit):
    html = client.get(f"/visit/{add_visit(species='Bushtit', raw_json=finder_raw('yolox', score=0.9))}").text
    tips = {m.group(1) for m in re.finditer(r'data-tip="([^"]*)"', html.replace("&#39;", "'"))}
    tight = next(t for t in tips if t.startswith("Tight look:"))
    assert "patch that moved" not in tight and "Motion detector" not in tight  # true on a YOLOX visit too
    assert "when the look falls short, Claude is asked" in tight
    html = client.get(f"/visit/{add_visit(model=classifier.MODEL_NAME, raw_json=finder_raw('blob', named=True))}").text
    page = html.replace("&#39;", "'")
    tips = {m.group(1) for m in re.finditer(r'data-tip="([^"]*)"', page)}
    circle = next(t for t in tips if t.startswith("Pi circle:"))
    assert "Motion detector" in circle and "crowd counter" in circle and "object detector" in circle
    assert "Claude wasn't asked" not in circle and "Claude was asked" in circle  # the tip also covers a miss
    assert "Circle from the Pi" in page and "Circle from the motion" not in page and pills_explained(html)


def test_a_bowl_named_visit_says_the_patch_was_pointed_out_without_a_tight_look(client, add_visit):
    bowl = {"calls": {"classifier": {**guess("Dark-eyed Junco"), "crop": "bowl"}, "skipped_claude": True,
                      "settled_by": "classifier", "circle": {**MOTION_CIRCLE, "pi": False},
                      "motion_circle": {"found": False, "why": "nothing", "source": "empty", "seconds": 0.02},
                      "finder": {"finder": "blob", "motion": {"found": False, "why": "nothing"},
                                 "tried": [{"finder": "motion", "found": False, "why": "nothing", "seconds": 0.1},
                                           {"finder": "blob", "found": True, "why": None, "seconds": 0.0}]}}}
    vid = add_visit(model=classifier.MODEL_NAME, raw_json=json.dumps(bowl))
    page = client.get(f"/visit/{vid}").text.replace("&#39;", "'")
    assert "The patch was pointed out by the crowd counter's outline; the circle shown is not the Pi's." in page
    assert "The patch for the tight look was pointed out" not in page
    # a tight miss keeps the fuller sentence
    raw = json.loads(finder_raw("blob", score=0.9))
    raw["calls"]["circle"] = {"x": 0.5, "y": 0.5, "r": 0.05, "claude": True}
    page = client.get(f"/visit/{add_visit(raw_json=json.dumps(raw))}").text.replace("&#39;", "'")
    assert "The patch for the tight look was pointed out by the crowd counter's outline; the circle shown is Claude's." in page


def test_about_page_mentions_the_second_photo_in_a_sentence(client):
    html = client.get("/about").text
    assert "second photo of the burst" in html and "only to keep score" in html
