"""Component tests of the web app through TestClient on a temp database: classifier."""

import pytest
import json

from helpers.classifier import MOTION_CIRCLE, MOTION_FOUND, guess, raw
from birdbrain import classifier, config


def test_visit_page_shows_the_guess_as_shadow(client, add_visit, monkeypatch):
    vid = add_visit(species="Bushtit", raw_json=json.dumps({"calls": {
        "haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}},
        "classifier": guess("Bushtit", "Dark-eyed Junco")}}))
    html = client.get(f"/visit/{vid}").text
    assert "BioCLIP 2 species check" in html and "Bushtit (50%), Dark-eyed Junco (50%)" in html
    assert 'class="badge shadow" data-tip="Scorecard:' in html and "Shadow:" not in html  # gate mode: no longer "shadow"
    monkeypatch.setattr(config, "CLASSIFIER", "shadow")
    assert 'class="badge shadow" data-tip="Shadow:' in client.get(f"/visit/{vid}").text
    assert "asked Sonnet" not in html  # the classifier step after Haiku isn't a second look


@pytest.mark.smoke
def test_stats_page_shows_the_scorecard(client, add_visit):
    assert "No visits scored yet" in client.get("/stats").text
    add_visit(species="Bushtit", raw_json=raw("Bushtit"))
    html = client.get("/stats").text
    assert 'id="classifier"' in html and "100%</span> first guess right (1/1)" in html


def test_visit_page_says_why_a_group_went_to_claude(client, add_visit):
    vid = add_visit(raw_json=json.dumps({"calls": {
        "classifier": guess("Dark-eyed Junco"), "gate_crowd": {"species": "Dark-eyed Junco", "animals": 3}}}))
    html = client.get(f"/visit/{vid}").text
    assert "the Pi saw 3 animals in view" in html


def test_visit_page_says_the_pi_named_it(client, add_visit):
    vid = add_visit(model="bioclip-2", raw_json=json.dumps({"calls": {
        "classifier": guess("Dark-eyed Junco"), "skipped_claude": True, "settled_by": "classifier"}}))
    html = client.get(f"/visit/{vid}").text
    assert "Claude wasn't asked" in html and 'data-tip="RPi:' in html


def tight_raw(why="score", name="Dark-eyed Junco", score=0.9, named=False, **first):
    """A visit's raw_json with a first look on the bowl, a tight look, and either Claude's answers with the
    scorecard look or (named) the tight look as the guess that named the visit."""
    tight = {"model": classifier.MODEL_NAME, "crop": "motion", "night": False, "seconds": 5.9,
             "top": [{"name": name, "scientific": "", "score": score}, {"name": "Bushtit", "scientific": "", "score": 0.05}],
             "local_share": 0.9, "circle": MOTION_CIRCLE}
    calls = json.loads(first_look_raw(**first))["calls"]
    if named:
        calls = {"classifier_gate": calls["classifier_gate"], "classifier": tight, "skipped_claude": True,
                 "settled_by": "classifier", "circle": MOTION_CIRCLE, "motion_circle": MOTION_FOUND}
    else:
        calls["classifier_tight"] = {**tight, "why": why}
        calls["sonnet"] = {"frames": 3, "answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}
    return json.dumps({"calls": calls})


def pills_explained(html):
    import re
    return html.count('class="badge ') == len(re.findall(r'class="badge [^"]*" data-tip=', html))


def test_visit_page_shows_the_tight_look_between_the_first_look_and_claude(client, add_visit):
    vid = add_visit(species="Bushtit", raw_json=tight_raw())
    html = client.get(f"/visit/{vid}").text
    steps = html[html.index("How it was decided"):]
    order = ["Black-capped Chickadee (61%)", "Dark-eyed Junco (90%)", "<strong>Haiku</strong>", "<strong>Sonnet</strong>",
             "Bushtit (50%)"]
    assert [steps.index(s) for s in order] == sorted(steps.index(s) for s in order)
    assert 'data-tip="First look:' in html and 'data-tip="Tight look:' in html and 'data-tip="Second look:' in html
    assert "Did not name the visit: its top guess was 61%, under the 80% it needs. The Pi took a tight look" in html
    assert "Did not name the visit either: its top guess was 90%, under the 95% it needs. Claude was asked." in html
    assert "on a tight square around the patch that moved, 5.9 s, before Claude" in html
    flow = html[html.index('aria-label="Detection path"'):]
    flow = flow[:flow.index("</ol>")]
    order = ["Black-capped Chickadee (61%)", "tight look", "Dark-eyed Junco (90%)", "Haiku", "Sonnet", "2nd look", "Bushtit (50%)"]
    assert [flow.index(s) for s in order] == sorted(flow.index(s) for s in order)
    assert pills_explained(html)
    # the What this photo trains box tells both looks apart
    assert ("but did not name the visit: its top guess was 61%, under the 80% it needs. Then a tight look at the patch "
            "that moved guessed Dark-eyed Junco, but its top guess was 90%, under the 95% it needs. Claude was asked.") in html


def test_visit_page_says_the_tight_look_named_it(client, add_visit):
    vid = add_visit(model="bioclip-2", raw_json=tight_raw(named=True, score=0.97))
    html = client.get(f"/visit/{vid}").text
    steps = html[html.index("How it was decided"):]
    assert steps.index("Black-capped Chickadee (61%)") < steps.index("Dark-eyed Junco (97%)")
    assert 'data-tip="RPi:' in html and 'data-tip="Tight look:' in html and 'data-tip="First look:' in html
    assert "Did not name the visit: its top guess was 61%, under the 80% it needs. The Pi took a tight look" in html
    assert "on a tight look around the patch that moved it was sure enough" in html and "Claude wasn't asked" in html
    assert "Named on the Pi, no circle" not in html
    assert ("Named the animal on the Pi:</b> Dark-eyed Junco (97%), on a tight look at the patch that moved after a "
            "first look at the whole bowl fell short.") in html
    assert "Second look:" not in html and "2nd look" not in html and pills_explained(html)
    flow = html[html.index('aria-label="Detection path"'):]
    flow = flow[:flow.index("</ol>")]
    assert flow.index("Black-capped Chickadee (61%)") < flow.index("tight look") < flow.index("Dark-eyed Junco (97%)")


def finder_raw(found, named, *tried, circle=None):
    """tight_raw plus the finder chain's record: `found` pointed at the patch (None: no finder did)."""
    import json
    raw = json.loads(tight_raw(named=named, score=0.97 if named else 0.9))
    calls = raw["calls"]
    calls["finder"] = {"finder": found, "motion": {"found": found == "motion", "why": None if found == "motion" else "nothing"},
                       "tried": [{"finder": n, "found": n == found, "why": None if n == found else ("no box" if n == "blob" else "nothing"),
                                  "seconds": 0.1} for n in tried]}
    key = "classifier" if named else "classifier_tight"
    calls[key]["finder"] = found
    if found != "motion":
        calls["motion_circle"] = {"found": False, "why": "nothing", "source": "empty", "seconds": 0.02}
        if circle and named:
            calls["circle"] = circle
    return json.dumps(raw)


def test_visit_page_names_the_finder_that_pointed_at_the_patch(client, add_visit):
    blob = {"x": 0.45, "y": 0.6, "r": 0.0562, "pi": True}
    vid = add_visit(model="bioclip-2", raw_json=finder_raw("blob", True, "motion", "blob", circle=blob))
    html = client.get(f"/visit/{vid}").text
    assert "on a tight square around the patch that moved (pointed out by the crowd counter&#39;s outline)" in html
    assert "Circle drawn on the Pi by the crowd counter&#39;s outline" in html and "the Motion detector found no patch" in html
    assert "No circle:" not in html
    vid = add_visit(raw_json=finder_raw("yolox", False, "motion", "blob", "yolox"))
    html = client.get(f"/visit/{vid}").text
    assert "patch that moved (pointed out by a small object detector (YOLOX-tiny)), 5.9 s, before Claude" in html
    assert "Did not name the visit either" in html
    vid = add_visit(model="bioclip-2", raw_json=finder_raw("motion", True, "motion"))
    html = client.get(f"/visit/{vid}").text
    assert "(pointed out by the Motion detector)" in html and "Circle drawn on the Pi from the motion" in html


def test_visit_page_lists_the_finders_tried_when_none_found_the_patch(client, add_visit):
    import json
    raw = json.loads(finder_raw(None, False, "motion", "blob", "yolox"))
    del raw["calls"]["classifier_tight"]  # no patch: no tight look was made
    vid = add_visit(raw_json=json.dumps(raw))
    html = client.get(f"/visit/{vid}").text
    assert "No circle: no clear patch of change near the zone that fired." in html
    assert ("data-tip=\"the Motion detector: no clear patch of change near the zone that fired; the crowd counter&#39;s outline: "
            "the crowd counter outlined no animal; a small object detector (YOLOX-tiny): saw no animal\"") in html
    assert "No other finder found the animal either" in html and "pointed out by" not in html


def test_visit_page_says_who_pointed_at_the_patch_when_the_circle_shown_is_claudes(client, add_visit):
    # The crowd counter's outline found the patch, the tight look missed, Claude answered and drew the circle: the
    # page must not say "no circle ... no other finder found the animal either" while the tight look's step says
    # the crowd counter's outline pointed it out.
    import json
    raw = json.loads(finder_raw("blob", False, "motion", "blob"))
    raw["calls"]["circle"] = {"x": 0.5, "y": 0.5, "r": 0.05, "claude": True}
    vid = add_visit(raw_json=json.dumps(raw))
    html = client.get(f"/visit/{vid}").text.replace("&#39;", "'")
    assert "The patch for the tight look was pointed out by the crowd counter's outline; the circle shown is Claude's." in html
    assert "(pointed out by the crowd counter's outline)" in html and "Did not name the visit either" in html
    assert "No circle:" not in html and "No other finder found the animal either" not in html and "found it" not in html
    # Claude drew no circle: the page says the Pi's circle was not kept, instead of claiming a circle it has not got.
    del raw["calls"]["circle"]
    html = client.get(f"/visit/{add_visit(raw_json=json.dumps(raw))}").text.replace("&#39;", "'")
    assert "pointed out by the crowd counter's outline; the Pi's circle was not kept." in html and "No circle:" not in html
    # The Motion detector found the patch and Claude drew the circle: the same sentence, naming the Motion detector.
    raw = json.loads(finder_raw("motion", False, "motion"))
    raw["calls"]["circle"] = {"x": 0.5, "y": 0.5, "r": 0.05, "claude": True}
    html = client.get(f"/visit/{add_visit(raw_json=json.dumps(raw))}").text.replace("&#39;", "'")
    assert "The patch for the tight look was pointed out by the Motion detector; the circle shown is Claude's." in html
    assert "No circle:" not in html and "found it" not in html


def test_visit_page_of_a_visit_circled_without_the_motion_detector(client, add_visit):
    # FINDERS left the Motion detector out (watch.MOTION_NOT_RUN): the circled step says it was not run, not that
    # it found no patch.
    import json
    blob = {"x": 0.45, "y": 0.6, "r": 0.0562, "pi": True}
    raw = json.loads(finder_raw("blob", True, "blob", circle=blob))
    raw["calls"]["motion_circle"] = {"found": False, "why": "not run", "source": None}
    del raw["calls"]["finder"]["motion"]
    html = client.get(f"/visit/{add_visit(model='bioclip-2', raw_json=json.dumps(raw))}").text.replace("&#39;", "'")
    assert "Circle drawn on the Pi by the crowd counter's outline" in html and ">(the Motion detector was not run)</span>." in html
    assert 'data-tip="The Motion detector: not run (FINDERS leaves the Motion detector out)."' in html
    assert "found no patch" not in html and "No circle:" not in html


def test_stats_page_reports_the_finders(client, add_visit):
    html = client.get("/stats").text
    assert 'id="finder"' in html and "No tight looks in the last 14 days yet" in html
    add_visit(model="bioclip-2", raw_json=finder_raw("blob", True, "motion", "blob", circle={"x": 0.5, "y": 0.5, "r": 0.05, "pi": True}))
    add_visit(raw_json=finder_raw("yolox", False, "motion", "blob", "yolox"))
    add_visit(raw_json=finder_raw("motion", False, "motion"))
    import json
    raw = json.loads(finder_raw(None, False, "motion", "blob", "yolox"))
    del raw["calls"]["classifier_tight"]
    add_visit(raw_json=json.dumps(raw))
    html = client.get("/stats").text
    assert "Tight looks in the last 14 days: 3" in html
    assert "the Motion detector on 1 (0 named); the crowd counter&#39;s outline on 1 (1 named); a small object detector (YOLOX-tiny) on 1 (0 named)." in html
    assert "No finder found a patch on 1 visit, so Claude was asked." in html


def test_visits_without_a_tight_look_render_as_before(client, add_visit):
    first_only = add_visit(species="Bushtit", raw_json=first_look_raw())
    html = client.get(f"/visit/{first_only}").text
    def no_tight_pill(html):  # neither the badge nor the Detection path's marker (the first look's tip may say the words)
        return ">tight look</span>" not in html and 'data-tip="Tight look:' not in html
    assert no_tight_pill(html) and "Did not name the visit either" not in html
    assert "Did not name the visit: its top guess was 61%, under the 80% it needs. Claude was asked." in html
    bowl_named = add_visit(model="bioclip-2", raw_json=json.dumps({"calls": {
        "classifier": guess("Dark-eyed Junco"), "skipped_claude": True, "settled_by": "classifier"}}))
    html = client.get(f"/visit/{bowl_named}").text
    assert no_tight_pill(html) and 'data-tip="RPi:' in html and "Named on the Pi, no circle" in html
    assert "<b>Named the animal on the Pi:</b> Dark-eyed Junco (50%). Claude was not asked." in html
    nothing = add_visit(species="Bushtit", raw_json=json.dumps({"calls": {
        "haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}}}))
    assert no_tight_pill(client.get(f"/visit/{nothing}").text)


def first_look_raw(why="score", name="Black-capped Chickadee", score=0.61, **extra):
    gate = {"model": classifier.MODEL_NAME, "crop": "bowl", "night": False, "seconds": 0.0, "why": why,
            "top": [{"name": name, "scientific": "", "score": score}, {"name": "Bushtit", "scientific": "", "score": 0.2}],
            "local_share": 0.9, **extra}
    return json.dumps({"calls": {"classifier_gate": gate, "classifier": guess("Bushtit"),
                                 "haiku": {"frames": 1, "answer": {"visitor": "bird", "species": "Bushtit",
                                                                   "confidence": "high"}}}})


def test_visit_page_shows_the_first_look_before_claude(client, add_visit):
    vid = add_visit(species="Bushtit", raw_json=first_look_raw())
    html = client.get(f"/visit/{vid}").text
    steps = html[html.index("How it was decided"):]
    assert steps.index("Black-capped Chickadee (61%)") < steps.index("<strong>Haiku</strong>") < steps.index("Bushtit (50%)")
    assert "Did not name the visit: its top guess was 61%, under the 80% it needs. Claude was asked." in html
    assert 'data-tip="First look:' in html and 'data-tip="Second look:' in html and "Shadow:" not in html
    flow = html[html.index('aria-label="Detection path"'):]
    flow = flow[:flow.index("</ol>")]
    assert flow.index("Black-capped Chickadee (61%)") < flow.index("Haiku") < flow.index("Bushtit (50%)")  # both pills
    assert "2nd look" in flow
    crowd = add_visit(raw_json=first_look_raw("crowd", "Dark-eyed Junco", 0.99))
    assert "more than one animal was in view" in client.get(f"/visit/{crowd}").text


def test_old_visit_without_a_first_look_renders_as_before(client, add_visit):
    vid = add_visit(species="Bushtit", raw_json=json.dumps({"calls": {
        "haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}},
        "classifier": guess("Bushtit")}}))
    html = client.get(f"/visit/{vid}").text
    assert "2nd look" not in html and "before Claude" not in html and "Its own guess, not used for the label" in html


def test_visit_settled_on_the_pi_never_says_claude_answered(client, add_visit):
    """Visit 995: named by the BioCLIP 2 species check, Claude not asked. The answer lines name who really answered."""
    note = "Named on the Pi by the BioCLIP 2 species check (99% sure); Claude was not asked."
    vid = add_visit(model="bioclip-2", notes=note, raw_json=tight_raw(named=True, score=0.97))
    html = client.get(f"/visit/{vid}").text
    assert "<dt>Species</dt>" not in html and "<dt>Also</dt>" not in html and "<dt>Notes</dt>" in html
    assert "species check said</dt>" not in html
    assert "<dt>Claude said</dt>" not in html and "<dt>Claude's notes</dt>" not in html
    assert '<dt>Identified by</dt><dd><span class="identified">BioCLIP 2 species check <span' in html
    assert "bioclip-2" not in html and "data-tip=\"Model:" not in html
    ident = html.split("<dt>Identified by</dt>")[1].split("</dl>")[0]
    assert ident.rstrip().endswith("</span></span></dd>") and "rpi-icon" in ident and "conf-high" in ident
    client.post(f"/visit/{vid}/correct", data={"visitor": "bird", "species": "House Finch"})
    assert "Corrected by you. BioCLIP 2 species check said: Dark-eyed Junco." in client.get(f"/visit/{vid}").text


def test_visit_answered_by_claude_still_says_claude(client, add_visit):
    vid = add_visit(notes="Looks like a junco.")
    html = client.get(f"/visit/{vid}").text
    assert "<dt>Claude said</dt>" not in html and ("<dt>Claude&#39;s notes</dt>" in html or "<dt>Claude's notes</dt>" in html)
    assert '<dt>Identified by</dt><dd><span class="identified"><span data-tip="Model: claude-haiku-5-5">Claude</span> <span' in html
    assert "claude-haiku-5-5 (escalated)" not in html and "(escalated)" not in html
    ident = html.split("<dt>Identified by</dt>")[1].split("</dl>")[0]
    assert ident.rstrip().endswith("</span></span></dd>") and "claude-mark" in ident


def test_escalated_claude_visit_names_sonnet_in_the_tip(client, add_visit):
    vid = add_visit(model="claude-sonnet-5-5", escalated=1)
    html = client.get(f"/visit/{vid}").text
    assert '<span data-tip="Model: claude-sonnet-5-5, escalated from Haiku">Claude</span>' in html
    assert "(escalated)" not in html and "claude-sonnet-5-5 " not in html.split("<dt>Identified by</dt>")[1].split("</dl>")[0]


def _conf_pill(html):
    import re
    return re.search(r'<span class="badge conf-high[^"]*"[^>]*>.*?</span>', html, re.S).group(0)


def test_confidence_pill_of_a_pi_visit_has_the_pi_icon_not_claude(client, add_visit):
    pi = add_visit(model="bioclip-2", raw_json=tight_raw(named=True, score=0.97))
    page = _conf_pill(client.get(f"/visit/{pi}").text)
    assert "rpi-icon" in page and "claude-mark" not in page and "by-claude" not in page
    assert "the BioCLIP 2 species check was very sure" in page and "Claude" not in page
    claude = add_visit()
    page = _conf_pill(client.get(f"/visit/{claude}").text)
    assert "claude-mark" in page and "by-claude" in page and "rpi-icon" not in page
    assert "Claude was sure" in page


def test_stats_page_shows_the_tight_look_bars(client, add_visit):
    html = client.get("/stats").text
    assert "Tight look bars (shadow)" in html and "No tight look to score yet" in html
    assert "<table class=\"classifier-species agree-bars\"" not in html
    add_visit(species="Dark-eyed Junco", raw_json=tight_raw(score=0.92, name="Black-capped Chickadee"))  # agrees, wrong
    add_visit(species="Bushtit", raw_json=tight_raw(score=0.96))  # disagrees, wrong
    html = client.get("/stats").text
    table = html[html.index("Tight look bars (shadow)"):]
    table = table[:table.index("</section>")]
    assert "<th>Bar</th>" in table and "Named when the bowl look agrees" in table and "Wrong when it agrees" in table
    assert table.index("<td>90%</td>") < table.index("<td>93%</td>") < table.index("<td>95%</td>")
    assert "<td>90%</td><td class=\"num\">2</td><td class=\"num\">2</td><td class=\"num\">1</td><td class=\"num\">1</td>" in table
    assert "<td>95%</td><td class=\"num\">1</td><td class=\"num\">1</td><td class=\"num\">0</td><td class=\"num\">0</td>" in table
    assert "changes nothing" in table and "settled-species rule is left out" in table
    assert "No tight look to score yet" not in table
