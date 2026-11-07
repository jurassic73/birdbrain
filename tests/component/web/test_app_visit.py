"""Component tests of the web app through TestClient on a temp database: visit."""

import io
import json
import re
from pathlib import Path

from PIL import Image

from helpers.web import calibrated, _keys_dialog
from birdbrain import config, db, images, zones
from birdbrain.web import app as webapp


def test_visit_page_and_404(client, add_visit):
    vid = add_visit(species="Song Sparrow")
    assert "Song Sparrow" in client.get(f"/visit/{vid}").text
    assert client.get("/visit/9999").status_code == 404


def test_visit_page_links_to_species_page_only_for_real_visits(client, add_visit):
    real = add_visit(species="Dark-eyed Junco")
    page = client.get(f"/visit/{real}").text
    assert 'id="species-page"' in page and 'href="/species/Dark-eyed%20Junco"' in page
    false = add_visit(visitor="none", species=None, confidence=None)
    assert 'id="species-page"' not in client.get(f"/visit/{false}").text


def test_false_trigger_popup_skips_confirmed_photos_and_lists_its_keys(client, add_visit, conn):
    old = add_visit(captured_at="2026-10-24T09:00:00", visitor="none", species=None)
    mid = add_visit(captured_at="2026-10-24T10:00:00", visitor="none", species=None)
    new = add_visit(captured_at="2026-10-24T11:00:00", visitor="none", species=None)
    db.confirm_visits(conn, [mid])
    html = client.get(f"/visit/{new}?ft=1&todo=1&embed=1").text
    assert f'href="/visit/{old}?ft=1&amp;todo=1&amp;embed=1" id="nav-next"' in html
    assert "<kbd>C</kbd> confirmed no animal" in html
    assert html.index('id="embed-close"') > html.index('embed-keys')  # Close sits at the far right, after the keys
    assert f'href="/visit/{mid}?ft=1" id="nav-next"' in client.get(f"/visit/{new}?ft=1").text  # without todo: all
    assert "Confirmed No Animal" in client.get(f"/visit/{mid}").text
    assert 'class="embed-keys' not in client.get(f"/visit/{mid}").text  # only the pop-up version lists keys


def test_false_trigger_page_shows_its_burst_in_order_and_delete_removes_it(client, stored_visit):
    vid = stored_visit(visitor="none", species=None, raw_json=json.dumps({"burst": {"count": 3, "best": 1, "fps": 4}}))
    with db.session() as conn:
        image_rel = db.get_visit(conn, vid)["image_path"]
    buf = io.BytesIO()
    Image.new("RGB", (640, 360)).save(buf, "JPEG")
    saved = images.store_burst([buf.getvalue()] * 3, image_rel)
    html = client.get(f"/visit/{vid}").text
    assert [html.index(p) for p in saved] == sorted(html.index(p) for p in saved)
    assert "+0.25 s" in html and "Claude saw" in html and "frame 2" in html
    assert 'data-tip="Previous frame (wraps)"' in html and 'data-tip="Next frame (wraps)"' in html and "/ 3" in html
    # Under the photo (inside its column, after the legend line), not in the details column; no on-photo hint.
    media = html[html.index('class="visit-media"'):html.index('class="viewer photo-viewer"')]
    assert 'class="vburst"' in media and media.index('id="mark-legend"') < media.index('class="vburst"')
    assert "photo-hint" not in html and "mark-hint" not in html
    client.post(f"/visit/{vid}/delete")
    assert images.burst_paths(image_rel) == []


def test_false_trigger_burst_has_a_play_button_left_of_the_arrows(client, burst_visit):
    vid, _, _ = burst_visit()
    html = client.get(f"/visit/{vid}").text
    assert 'id="vburst-play"' in html and 'aria-pressed="false"' in html
    assert html.index('id="vburst-play"') < html.index('data-step="-1"')
    # Speed toggle: one button right after play, before the arrows; the server always renders 1x.
    assert html.count('id="vburst-speed"') == 1
    tag = re.search(r'<button[^>]*id="vburst-speed"[^>]*>(.*?)</button>', html, re.S)
    assert tag, "speed button missing"
    attrs = tag.group(0)[:tag.group(0).index(">") + 1]
    classes = re.search(r'class="([^"]*)"', attrs).group(1).split()
    assert "vburst-step" in classes and "vburst-speed" in classes
    assert 'data-speed="1"' in attrs and tag.group(1).strip() == "1x"
    assert re.search(r'aria-label="[^"]+"', attrs) and re.search(r'data-tip="[^"]+"', attrs)
    assert html.index('id="vburst-play"') < html.index('id="vburst-speed"') < html.index('data-step="-1"')


def test_burst_arrows_absent_without_a_burst(client, stored_visit):
    vid = stored_visit(visitor="none", species=None)
    assert 'id="vburst-play"' not in client.get(f"/visit/{vid}").text
    assert 'id="vburst-speed"' not in client.get(f"/visit/{vid}").text
    assert 'class="vburst-step"' not in client.get(f"/visit/{vid}").text


def test_picking_a_burst_frame_as_the_photo_is_admin_only_and_false_triggers_only(client, burst_visit, stored_visit, monkeypatch):
    vid, image_rel, saved = burst_visit()
    real = stored_visit()
    assert client.post(f"/visit/{real}/photo-frame", data={"frame": "1"}).status_code == 400  # one frame only
    for bad in ("5", "-1", "x", "99"):
        assert client.post(f"/visit/{vid}/photo-frame", data={"frame": bad}).status_code == 400, bad
    assert client.post(f"/visit/{vid}/photo-frame", data={"frame": "1"}).headers["HX-Refresh"] == "true"
    assert client.post("/visit/999999/photo-frame", data={"frame": "1"}).status_code == 404
    # Visitors: the change is refused, and a false trigger's page isn't theirs to see at all.
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    assert client.post(f"/visit/{vid}/photo-frame", data={"frame": "0"}).status_code == 403
    assert client.get(f"/visit/{vid}").status_code == 404
    with db.session() as conn:
        assert db.get_visit(conn, vid)["photo_frame"] == 1


def test_picked_frame_becomes_the_photo_on_the_page_and_the_card_and_reset_restores_claudes(client, burst_visit, conn):
    vid, image_rel, saved = burst_visit()
    thumb = images.burst_thumb_path(image_rel, 3)
    html = client.get(f"/visit/{vid}").text
    assert f'src="/images/{image_rel}"' in html and 'data-frame=""' in html
    assert 'id="use-frame" hidden' in html and 'id="use-claude"' not in html  # nothing to go back to yet
    assert 'class="vburst-frame on" data-src="/images/' + saved[1] in html  # Claude's frame is the one shown
    client.post(f"/visit/{vid}/photo-frame", data={"frame": "3"})
    v = db.get_visit(conn, vid)
    assert v["photo_frame"] == 3 and v["image_path"] == image_rel  # Claude's frame stays the record
    assert (config.IMAGES_DIR / thumb).exists() and images.burst_paths(image_rel) == saved
    assert db.photo_rel(v) == saved[2] and db.thumb_rel(v) == thumb
    html = client.get(f"/visit/{vid}").text
    assert f'src="/images/{saved[2]}"' in html and f'href="/images/{saved[2]}"' in html and 'data-frame="3"' in html
    assert 'class="vburst-frame on" data-src="/images/' + saved[2] in html and html.count('vburst-frame on"') == 1
    assert "your photo" in html and "frame 3" in html and "Claude saw" in html and "frame 2" in html
    assert 'id="use-claude"' in html
    card = client.get("/false-triggers").text
    assert f'src="/images/{thumb}"' in card and f'data-full="/images/{saved[2]}"' in card
    assert f'src="/images/{v["thumb_path"]}"' not in card
    # Picking the frame Claude saw is the same as no pick.
    client.post(f"/visit/{vid}/photo-frame", data={"frame": "2"})
    assert db.get_visit(conn, vid)["photo_frame"] is None and not (config.IMAGES_DIR / thumb).exists()
    client.post(f"/visit/{vid}/photo-frame", data={"frame": "4"})
    client.post(f"/visit/{vid}/photo-frame", data={"frame": ""})  # reset
    v = db.get_visit(conn, vid)
    assert v["photo_frame"] is None and images.burst_thumbs(image_rel) == []
    assert db.photo_rel(v) == image_rel and db.thumb_rel(v) == v["thumb_path"]
    assert f'src="/images/{v["thumb_path"]}"' in client.get("/false-triggers").text
    assert images.burst_paths(image_rel) == saved and all((config.IMAGES_DIR / r).exists() for r in (image_rel, *saved))
    # Once the photo budget has left only the thumbnail, nothing can be picked.
    conn.execute("UPDATE visits SET photo_level = 'thumb' WHERE id = ?", (vid,))
    conn.commit()
    assert client.post(f"/visit/{vid}/photo-frame", data={"frame": "3"}).status_code == 400


def test_circles_stay_on_their_frame_when_another_frame_becomes_the_photo(client, burst_visit, conn):
    from birdbrain import presence
    vid, image_rel, saved = burst_visit()
    on_claudes = {"x": 0.5, "y": 0.5, "r": 0.05}
    on_three = {"x": 0.2, "y": 0.3, "r": 0.05, "frame": 3}
    client.post(f"/visit/{vid}/correct", data={"visitor": "none", "feedback": "1 is a leaf, 2 is a leaf",
                                               "marks": json.dumps([on_claudes, on_three])})
    photo_marks = webapp.templates.env.globals["photo_marks"]
    assert photo_marks(db.get_visit(conn, vid)) == [on_claudes]
    client.post(f"/visit/{vid}/photo-frame", data={"frame": "3"})
    v = db.get_visit(conn, vid)
    # Stored as drawn; the card (frame 3 now) shows only frame 3's circle; Claude's frame is still frame 2.
    assert db.parse_marks(v["marks"]) == [on_claudes, on_three]
    assert photo_marks(v) == [on_three] and db.claude_frame(v["raw_json"]) == 2
    assert presence.owner_frame(v["marks"], json.loads(v["raw_json"])) is None  # the first circle is on Claude's photo
    html = client.get(f"/visit/{vid}").text
    assert "&#34;frame&#34;: 3" in html and html.count("&#34;x&#34;: ") == 2
    assert "(on burst frame 3, not the photo you saw)" in db.recent_lessons(conn)[0]
    # Saving again while frame 3 is the photo: a circle on Claude's frame (2) still folds to "no frame".
    client.post(f"/visit/{vid}/correct", data={"visitor": "none", "feedback": "leaves",
                                               "marks": json.dumps([{**on_three}, {**on_claudes, "frame": 2}])})
    assert db.parse_marks(db.get_visit(conn, vid)["marks"]) == [on_three, on_claudes]
    client.post(f"/visit/{vid}/photo-frame", data={"frame": "0"})
    assert photo_marks(db.get_visit(conn, vid)) == [on_claudes]


def test_deleting_a_visit_with_a_picked_frame_removes_every_file_once(client, burst_visit):
    vid, image_rel, saved = burst_visit()
    client.post(f"/visit/{vid}/photo-frame", data={"frame": "4"})
    folder = (config.IMAGES_DIR / image_rel).parent
    stem = Path(image_rel).stem
    assert sorted(p.name for p in folder.iterdir() if p.name.startswith(stem)) == sorted(
        [f"{stem}.jpg", f"{stem}_t.jpg", f"{stem}_b4_t.jpg"] + [f"{stem}_b{n}.jpg" for n in range(1, 5)])
    client.post(f"/visit/{vid}/delete")
    assert not [p for p in folder.iterdir() if p.name.startswith(stem)]


def test_marking_no_animal_moves_the_visit_to_false_triggers_and_teaches_claude(client, add_visit, conn):
    vid = add_visit(needs_review=1)
    r = client.post(f"/visit/{vid}/correct", data={"visitor": "none", "species": "Dark-eyed Junco", "feedback": "rock tip"})
    assert r.headers["HX-Refresh"] == "true"
    v = db.get_visit(conn, vid)
    assert (v["corrected_visitor"], v["corrected_species"], v["needs_review"]) == ("none", None, 0)
    assert f"/visit/{vid}" in client.get("/false-triggers?show=all").text  # corrected counts as confirmed
    assert "rock tip" in db.recent_lessons(conn)[0]
    client.post(f"/visit/{vid}/clear-correction")
    assert db.get_visit(conn, vid)["corrected_visitor"] is None


def test_feedback_circles_are_saved_shown_and_described_to_claude(client, add_visit, conn):
    vid = add_visit()
    marks = json.dumps([{"x": 0.8, "y": 0.75, "r": 0.05}, {"x": "bad"}, {"x": 0.5, "y": 0.3, "r": 0.1}])
    client.post(f"/visit/{vid}/correct", data={"visitor": "none", "feedback": "1 is a leaf, 2 is a rock", "marks": marks})
    assert db.parse_marks(db.get_visit(conn, vid)["marks"]) == [{"x": 0.8, "y": 0.75, "r": 0.05}, {"x": 0.5, "y": 0.3, "r": 0.1}]
    assert "circle 1 at 80% across, 75% down; circle 2 at 50% across, 30% down" in db.recent_lessons(conn)[0]
    assert '&#34;x&#34;: 0.8, &#34;y&#34;: 0.75' in client.get(f"/visit/{vid}").text
    client.post(f"/visit/{vid}/clear-correction")
    assert db.get_visit(conn, vid)["marks"] is None


def test_saving_circles_on_burst_frames_keeps_them_off_the_cards(client, add_visit, conn):
    vid = add_visit(raw_json=json.dumps({"burst": {"count": 6, "best": 2, "fps": 4}}))
    marks = [{"x": 0.4, "y": 0.4, "r": 0.05, "frame": 3}, {"x": 0.6, "y": 0.6, "r": 0.05, "frame": 5}]
    client.post(f"/visit/{vid}/correct", data={"visitor": "bird", "marks": json.dumps(marks)})
    # frame 3 is the photo Claude saw, so that circle needs no frame
    assert db.parse_marks(db.get_visit(conn, vid)["marks"]) == [{"x": 0.4, "y": 0.4, "r": 0.05},
                                                                {"x": 0.6, "y": 0.6, "r": 0.05, "frame": 5}]
    card = webapp.templates.env.globals["photo_marks"](db.get_visit(conn, vid))
    assert card == [{"x": 0.4, "y": 0.4, "r": 0.05}]


def test_visit_page_starts_with_claudes_circle_until_the_owner_saves_their_own(client, add_visit, conn):
    circle = {"x": 0.45, "y": 0.65, "r": 0.05, "claude": True}
    vid = add_visit(raw_json=json.dumps({"calls": {"circle": circle}}))
    assert "&#34;claude&#34;: true" in client.get(f"/visit/{vid}").text
    client.post(f"/visit/{vid}/correct", data={"feedback": "1 is the tail", "marks": json.dumps([circle])})
    assert "circle 1 (the one you drew) at 45% across, 65% down" in db.recent_lessons(conn)[0]
    db.correct_visit(conn, vid, None, None, "note", [{"x": 0.1, "y": 0.1, "r": 0.05}])
    assert "claude" not in client.get(f"/visit/{vid}").text.split('data-marks="')[1].split('"')[0]


ESCALATED = json.dumps({"calls": {
    "haiku": {"model": "claude-haiku-5-5", "answer": {"visitor": "bird", "species": "Dark-eyed Junco",
                                                      "confidence": "medium", "notes": "head hidden"}},
    "sonnet": {"model": "claude-sonnet-5-5", "frames": 3, "answer": {"visitor": "bird", "species": "Song Sparrow",
                                                                     "confidence": "high", "notes": "streaked breast"}},
    "lessons": 4,
}})


def test_visit_page_shows_how_claude_decided(client, add_visit):
    vid = add_visit(species="Song Sparrow", model="claude-sonnet-5-5", escalated=1, raw_json=ESCALATED)
    html = client.get(f"/visit/{vid}").text
    assert "How it was decided" in html
    assert "looked at 1 frame (the sharpest of the burst)" in html and "looked at 3 frames" in html
    assert "head hidden" in html and "streaked breast" in html and "second look" in html
    assert "Used 4 of your past corrections" in html


def test_visit_page_says_when_claude_could_not_answer(client, add_visit):
    raw = json.dumps({"answer": {"visitor": "other"}, "calls": {"error": "APIConnectionError: Connection error."},
                      "trigger": "rim"})
    vid = add_visit(visitor="other", species=None, confidence="low", needs_review=1, raw_json=raw)
    decision = webapp.decision_steps(raw)
    assert decision["failed"] and [(s["model"], s["error"]) for s in decision["steps"]] == [
        ("Haiku", "APIConnectionError: Connection error.")]
    html = client.get(f"/visit/{vid}").text
    assert "Couldn't answer: the call failed." in html and "APIConnectionError" in html
    assert "Claude couldn't answer, so the photo was kept and sent to you for review." in html
    assert "Declined to answer" not in html
    assert f"/visit/{vid}" in client.get("/review").text
    assert webapp.decision_steps(None)["steps"] == []
    assert webapp.decision_steps('{"answer": {}}')["steps"] == []   # dev-seed rows have no trail
    assert webapp.decision_steps("not json")["steps"] == []


def test_visit_page_shows_the_local_detector_step(client, add_visit):
    gated = json.dumps({"calls": {"detector": {"animal": False, "best": None, "seconds": 0.41}, "skipped_claude": True}})
    vid = add_visit(visitor="none", species=None, model="yolox-tiny", raw_json=gated)
    html = client.get(f"/visit/{vid}").text
    assert "Local detector" in html and "0.41 s" in html and "Claude wasn't asked" in html
    shadow = json.dumps({"calls": {"detector": {"animal": True, "best": {"label": "bird", "score": 0.62}, "seconds": 0.4},
                                   "haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}}})
    html = client.get(f"/visit/{add_visit(species='Bushtit', raw_json=shadow)}").text
    assert "bird (62%)" in html and "Bushtit" in html and "Claude wasn't asked" not in html


def test_visit_without_a_trail_hides_the_section(client, add_visit):
    assert "How it was decided" not in client.get(f"/visit/{add_visit()}").text


def test_correct_rejects_unknown_visitor_and_missing_visit(client, add_visit):
    vid = add_visit()
    assert client.post(f"/visit/{vid}/correct", data={"visitor": "dragon"}).status_code == 400
    assert client.post("/visit/9999/correct", data={"visitor": "none"}).status_code == 404


def test_delete_removes_row_and_image_files(client, stored_visit, conn):
    vid = stored_visit()
    row = db.get_visit(conn, vid)
    r = client.post(f"/visit/{vid}/delete")
    assert r.headers["HX-Redirect"] == "/"
    assert db.get_visit(conn, vid) is None
    assert not (config.IMAGES_DIR / row["image_path"]).exists()
    assert not (config.IMAGES_DIR / row["thumb_path"]).exists()


def view_rect(z, trigger):
    """The dashed box's SVG attributes for the Pi animal check's view (classifier.presence_view_box)."""
    from birdbrain import classifier
    r = webapp.box_rect(classifier.presence_view_box(z, trigger, webapp.ZONE_W / webapp.ZONE_H))
    return f'x="{round(r["x"], 2)}" y="{round(r["y"], 2)}" width="{round(r["w"], 2)}" height="{round(r["h"], 2)}"'


def test_visit_photo_draws_its_crop_box_and_the_zone_that_fired(client, add_visit, calibrated, monkeypatch):
    monkeypatch.setattr(config, "GROUND_ZONE", ((0.1, 0.5), (0.9, 0.5), (0.9, 1), (0.1, 1)))
    monkeypatch.setattr(config, "GROUND_CROP", (0.0, 0.4, 1.0, 0.6))
    now = zones.from_env()
    colors = webapp.zone_colors(now)["colors"]
    # A rim trigger: the rim band and the rock in their Settings colors, the bath crop dashed in the crop's color.
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": now.to_dict()}))
    html = client.get(f"/visit/{vid}").text
    assert 'class="zones-overlay"' in html and html.count('class="vz-zone"') == 2
    assert f'class="vz-zone" style="color: {colors["ring"]}"' in html
    assert f'class="vz-crop" style="color: {colors["crop"]}" {view_rect(now, "rim")}' in html
    assert "the rim and rocks, the zone whose motion took this photo" in html and "today&#39;s zones" not in html
    # Water: the open water. Ground: the ground polygon and the ground crop, in the ground's color.
    vid = add_visit(raw_json=json.dumps({"trigger": "water", "zones": now.to_dict()}))
    html = client.get(f"/visit/{vid}").text
    assert html.count('class="vz-zone"') == 1 and f'class="vz-zone" style="color: {colors["water"]}"' in html
    vid = add_visit(visitor="none", species=None, raw_json=json.dumps({"trigger": "ground", "zones": now.to_dict()}))
    html = client.get(f"/visit/{vid}").text
    assert f'class="vz-zone" style="color: {colors["ground"]}"' in html
    assert f'class="vz-crop" style="color: {colors["ground"]}" x="0.0" y="36.0" width="160.0" height="54.0"' in html


def test_visit_photo_uses_the_zones_it_was_taken_with(client, add_visit, calibrated, conn, monkeypatch):
    then = zones.from_env()
    monkeypatch.setattr(config, "CROP_ROI", (0.0, 0.0, 0.5, 0.5))  # recalibrated since
    monkeypatch.setattr(config, "MOTION_RING", (0.3, 0.5, 0.15, 0.2, 0.1, 0.15, 0.5))  # the bowl view follows the ring
    today = zones.from_env()
    assert view_rect(then, "rim") != view_rect(today, "rim")
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": then.to_dict()}))
    assert view_rect(then, "rim") in client.get(f"/visit/{vid}").text
    # An older visit (no zones stored) with no calibration history is drawn with today's, and says so.
    vid = add_visit(raw_json=json.dumps({"trigger": "rim"}), captured_at="2026-10-20T12:00:00")
    html = client.get(f"/visit/{vid}").text
    assert view_rect(today, "rim") in html and "today&#39;s zones" in html
    # With a history, the zones in use at the time: the last run before it, else the first run's before.
    moved = zones.Zones(crop=(0.5, 0.5, 0.25, 0.25))
    db.add_calibration(conn, ran_at="2026-10-21T08:00:00", reason="hand", status="moved", summary="",
                       before_json=json.dumps(then.to_dict()), after_json=json.dumps(moved.to_dict()))
    html = client.get(f"/visit/{vid}").text
    assert view_rect(then, "rim") in html and "calibration history" in html
    vid = add_visit(raw_json=json.dumps({"trigger": "rim"}), captured_at="2026-10-22T12:00:00")
    assert 'x="80.0" y="45.0" width="40.0" height="22.5"' in client.get(f"/visit/{vid}").text
    # A capture with no trigger (watch --once): only the crop box.
    vid = add_visit(raw_json=json.dumps({}))
    html = client.get(f"/visit/{vid}").text
    assert 'class="vz-crop"' in html and 'class="vz-zone"' not in html and "Solid outline" not in html


def test_visit_photo_without_any_zones_draws_nothing(client, add_visit, monkeypatch):
    monkeypatch.setattr(config, "MOTION_RING", None)
    monkeypatch.setattr(config, "MOTION_ROI", None)
    monkeypatch.setattr(config, "CROP_ROI", None)
    vid = add_visit(raw_json=json.dumps({"trigger": "rim"}))
    html = client.get(f"/visit/{vid}").text
    assert 'class="zones-overlay"' not in html and 'id="zones-toggle"' not in html


def test_visit_page_says_which_zone_saw_the_motion(client, add_visit):
    vid = add_visit(species="Virginia Opossum", visitor="possum", raw_json=json.dumps({"trigger": "ground", "calls": {
        "haiku": {"answer": {"visitor": "possum", "species": "Virginia Opossum", "confidence": "high"}}}}))
    assert "Spotted moving on the ground." in client.get(f"/visit/{vid}").text
    assert webapp.decision_steps(json.dumps({"calls": {}}))["trigger"] is None   # --once captures have none


def test_lesson_edit_changes_label_and_note_but_keeps_circles(client, add_visit, conn):
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    client.post(f"/visit/{vid}/correct", data={"visitor": "none", "feedback": "rock tip", "marks": '[{"x":0.5,"y":0.5,"r":0.1}]'})
    assert f'hx-post="/lessons/{vid}/edit"' in client.get("/notes").text
    client.post(f"/lessons/{vid}/edit", data={"visitor": "bird", "species": "Bushtit", "feedback": "a bushtit on the rocks"})
    line = db.recent_lessons(conn)[0]
    assert '"Bushtit"' in line and "a bushtit on the rocks" in line and "circle 1 at 50% across" in line
    assert client.post(f"/lessons/{vid}/edit", data={"visitor": "dragon"}).status_code == 400


def test_species_search_ranks_ebird_names_and_skips_known(client):
    from birdbrain import names
    assert names.search("house f", []) == ["House Finch"]
    assert names.search("housefinch", []) == ["House Finch"]
    assert names.search("finch", [])[0] == "House Finch"  # word match, shortest first
    assert names.search("fin hou", []) == ["House Finch"]
    assert "Bushtit" not in client.get("/species-search", params={"q": "bush", "known": "Bushtit"}).json()
    assert client.get("/species-search", params={"q": "h"}).json() == []
    assert names.canonical("house-finch") == "House Finch" and names.canonical("Western House Finch") is None


def test_ask_claude_names_the_species_from_a_clue(client, conn, stored_visit, monkeypatch):
    from birdbrain import names
    vid = stored_visit(species="Dark-eyed Junco")
    seen = {}

    def ask(jpeg_b64, hint, captured_at, location=""):
        seen["hint"] = hint
        return names.Suggestion(name="house finch", visitor="bird", confidence="high", reason="Red crown and breast."), \
            {"input_tokens": 2000, "output_tokens": 100}
    monkeypatch.setattr(names, "ask", ask)
    r = client.post(f"/visit/{vid}/ask-species", data={"hint": "red head finch"}).json()
    assert seen["hint"] == "red head finch"
    assert r["name"] == "House Finch" and r["visitor"] == "bird" and r["usd"] > 0  # official spelling
    assert db.get_visit(conn, vid)["corrected_species"] is None  # only a suggestion until saved
    monkeypatch.setattr(names, "ask", lambda *a, **k: (names.Suggestion(
        name="Western House Finch", visitor="bird", confidence="high", reason="x"), {}))
    assert "isn't an eBird species name" in client.post(f"/visit/{vid}/ask-species", data={"hint": "x"}).json()["error"]
    assert "clue" in client.post(f"/visit/{vid}/ask-species", data={"hint": " "}).json()["error"]
    html = client.get(f"/visit/{vid}").text
    assert "ask-btn" in html and "claude-mark" in html


def test_false_trigger_opens_in_a_pop_up_that_keeps_its_arrows_inside(client, add_visit):
    older = add_visit(visitor="none", species=None, captured_at="2026-10-31T08:00:00")
    vid = add_visit(visitor="none", species=None, captured_at="2026-10-31T09:00:00")
    html = client.get(f"/visit/{vid}?ft=1&embed=1").text
    assert '<header class="topbar">' not in html and "/static/tab-alert.js" not in html and 'class="embed"' in html
    assert f'href="/visit/{older}?ft=1&amp;embed=1"' in html and 'id="embed-close"' in html
    normal = client.get(f"/visit/{vid}?ft=1").text
    assert '<header class="topbar">' in normal and "embed=1" not in normal and 'id="embed-close"' not in normal


def test_visit_page_says_notes_are_for_claude_only(client, add_visit):
    html = client.get(f"/visit/{add_visit()}").text
    # The note box is one short line; its info tip says the Pi never reads notes.
    assert "Note for Claude" in html and "The Pi animal check never reads notes" in html
    assert 'id="feedback" name="feedback" rows="1"' in html
    assert "The answer you pick trains the Pi animal check" in html


def _flows(html):
    return re.findall(r'<ol class="decision-flow.*?</ol>', html, re.S)


def test_detection_path_strip_is_labelled_and_marks_claude_pills(client, add_visit):
    flows = _flows(client.get(f"/visit/{add_visit(species='Song Sparrow', model='claude-sonnet-5-5', escalated=1, raw_json=ESCALATED)}").text)
    assert len(flows) == 2  # the nav bar and the phone copy
    for f in flows:
        assert 'aria-label="Detection path"' in f and ">Detection path</li>" in f
        pills = re.findall(r'<li class="flow-pill[^"]*">(.*?)</li>', f, re.S)
        for name in ("Haiku", "Sonnet"):
            assert all("claude-mark" in p for p in pills if f"<strong>{name}</strong>" in p)
            assert any(f"<strong>{name}</strong>" in p for p in pills)
        assert "claude-mark" not in next(p for p in pills if p.startswith("Motion"))
        assert "claude-mark" not in pills[-1]


def test_local_only_steps_get_no_claude_mark(client, add_visit):
    raw = json.dumps({"calls": {"presence": {"empty": True, "score": 0.02, "threshold": 0.5, "trained_on": 40}, "skipped_claude": True}})
    html = client.get(f"/visit/{add_visit(visitor='none', species=None, model='presence', raw_json=raw)}").text
    flows = _flows(html)
    assert flows and all("Detection path" in f and "Pi animal check" in f and "claude-mark" not in f for f in flows)


def test_note_for_claude_label_has_the_mark(client, add_visit):
    html = client.get(f"/visit/{add_visit()}").text
    assert re.search(r'<label for="feedback" class="tip-label"><svg class="claude-mark".*?</svg>Note for Claude', html, re.S)


def test_false_trigger_page_moves_on_after_save_but_normal_visits_reload(client, add_visit):
    vid = add_visit(visitor="none", species=None)
    bird = add_visit(visitor="bird", species="Bushtit")
    ft = client.get(f"/visit/{vid}?ft=1").text
    assert 'location.href = next ? next.href : "/false-triggers"' in ft
    assert "check-save" in ft
    assert 'location.href = next ?' not in client.get(f"/visit/{bird}").text


def test_c_key_confirms_and_moves_on_for_admin_visit_pages(client, add_visit, monkeypatch):
    vid = add_visit(visitor="none", species=None)
    html = client.get(f"/visit/{vid}").text
    assert "// C: confirm this visit" in html and "Press C to confirm and go to the next older visit." in html
    assert "// C: confirmed no animal" not in html
    embed = client.get(f"/visit/{vid}?embed=1").text
    assert "// C: confirm this visit" not in embed and "// C: confirmed no animal" in embed
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    assert "// C: confirm this visit" not in client.get(f"/visit/{vid}").text


def test_visit_page_keeps_visits_nav_highlighted(client, add_visit):
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    page = client.get(f"/visit/{vid}").text
    assert re.search(r'<a href="/" aria-current="page">Visits</a>', page)
    assert 'href="/visits"' not in page


def test_review_heading_shows_the_real_total(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "REVIEW_SHOWN", 2)
    for i in range(3):
        add_visit(needs_review=1, captured_at=f"2026-11-01T10:00:{i:02d}")
    html = client.get("/review").text
    assert 'Review queue <span class="muted">3</span>' in html and html.count('class="card"') == 2
    assert "Showing the first 2 of 3" in html


def test_embedded_visit_page_does_not_swipe(client, add_visit):
    vid = add_visit()
    html = client.get(f"/visit/{vid}?embed=1").text
    assert "data-swipe=" not in html and "swipe.js" not in html


def test_visit_page_back_link_replaces_all_visits(client, add_visit):
    vid = add_visit(species="Bushtit")
    html = client.get(f"/visit/{vid}").text
    assert "All visits" not in html
    assert html.count('href="/" data-back-home>') == 2 and "<span>Back</span>" in html


def test_confidence_pills_carry_the_claude_mark(client, add_visit):
    """A high/medium/low pill is Claude's answer, so it gets the mark; Needs review reuses conf-low without it."""
    vid = add_visit(confidence="low", needs_review=1)
    html = client.get(f"/visit/{vid}").text
    assert re.search(r'class="badge conf-low by-claude"[^>]*><svg class="claude-mark"', html)
    assert re.search(r'class="badge conf-low"[^>]*>Needs review', html)


def test_p_key_plays_the_burst_and_is_listed_only_with_a_burst(client, burst_visit, stored_visit):
    vid, _, _ = burst_visit()
    html = client.get(f"/visit/{vid}").text
    rows, _ = _keys_dialog(html)
    assert ("P", "Play or stop the burst") in rows
    assert "P)" in html and "e.key.toLowerCase() !== \"p\"" in html
    emb = client.get(f"/visit/{vid}?ft=1&embed=1").text
    assert "birdbrain:play" in emb and "<kbd>P</kbd> play burst" in emb
    assert "birdbrain:play" in client.get("/false-triggers").text
    plain = stored_visit(visitor="none", species=None)
    assert ("P", "Play or stop the burst") not in _keys_dialog(client.get(f"/visit/{plain}").text)[0]
