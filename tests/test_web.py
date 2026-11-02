import io
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from birdbrain import alert, config, db, images, zones
from birdbrain.web import app as webapp


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def stored_visit(add_visit):
    """A visit with real image files on disk, so delete can be checked."""
    def add(**fields) -> int:
        buf = io.BytesIO()
        Image.new("RGB", (640, 360), (90, 120, 200)).save(buf, "JPEG")
        image_rel, thumb_rel = images.store_frame(buf.getvalue(), datetime.now())
        return add_visit(image_path=image_rel, thumb_path=thumb_rel, **fields)
    return add


def test_home_empty_state_hides_filters(client):
    html = client.get("/").text
    assert "No visitors yet." in html and "more-filters" not in html


def test_home_lists_visits_with_chips_and_last_motion(client, add_visit):
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    html = client.get("/").text
    assert "Bushtit" in html and "tally-box" in html and "Mammals" in html
    assert "Last motion" in html


def test_home_lists_todays_visitors_with_counts(client, add_visit):
    assert "None yet today." in client.get("/").text
    add_visit(species="Bushtit")
    add_visit(species="Bushtit")
    add_visit(visitor="squirrel", species=None)
    add_visit(visitor="none", species=None)
    critters = client.get("/").text.split('class="critters"')[1].split("</ul>")[0]
    assert critters.index("Bushtit") < critters.index("Squirrel")
    assert ">2</a>" in critters and ">1</a>" in critters and "None" not in critters


def test_scoreboard_periods(client, add_visit):
    add_visit(species="Bushtit", captured_at=(datetime.now() - timedelta(days=3)).replace(microsecond=0).isoformat())
    assert "None yet today." in client.get("/scoreboard?range=today").text
    assert "None yesterday." in client.get("/scoreboard?range=yesterday").text
    week = client.get("/scoreboard?range=week").text
    assert "Visitors this week" in week and "Bushtit" in week and 'class="score-range on"' in week
    assert "Visitors today" in client.get("/scoreboard?range=bogus").text
    assert "Bushtit" in client.get("/scoreboard?range=all").text
    # The This week chip shows the week on the scoreboard too.
    href = next(f for n, f in webapp.quick_filters() if n == "This week")
    assert "Visitors this week" in client.get("/?" + webapp.query_string(href)).text


def test_home_filter_chip_is_marked_active(client, add_visit):
    add_visit(visitor="raccoon", species=None)
    html = client.get("/?kind=mammal").text
    assert 'aria-current="page">Mammals<' in html and "matching</h2>" in html


def test_load_more_pages_in_24s(client, add_visit):
    for _ in range(30):
        add_visit()
    first = client.get("/").text
    assert first.count('class="card"') == 24 and "/visits/more?offset=24" in first
    more = client.get("/visits/more?offset=24").text
    assert more.count('class="card"') == 6 and "load-more" not in more


def test_home_splits_cards_into_today_and_earlier(client, add_visit):
    old = (datetime.now() - timedelta(days=2)).replace(microsecond=0).isoformat()
    for _ in range(20):
        add_visit(captured_at=old)
    for _ in range(26):
        add_visit()
    first = client.get("/").text
    assert first.count("grid-break grid-today") == 1 and "grid-break grid-earlier" not in first  # 24 cards, all from today
    more = client.get("/visits/more?offset=24").text
    assert "grid-break grid-today" not in more and more.count("grid-break grid-earlier") == 1
    assert more.index("grid-break grid-earlier") > more.index('class="card"')  # 2 more today first, then Earlier
    assert "grid-break grid-earlier" not in client.get("/visits/more?offset=48").text
    assert "grid-break grid-today" not in client.get("/?kind=bird").text  # filters show one plain grid


def test_home_says_no_visitors_yet_today(client, add_visit):
    add_visit(captured_at=(datetime.now() - timedelta(days=1)).replace(microsecond=0).isoformat())
    page = client.get("/").text
    assert "No visitors yet today." in page and page.index("grid-break grid-today") < page.index("grid-break grid-earlier")


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


def test_every_status_pill_explains_itself(client, add_visit):
    add_visit(visitor="none", species=None, confidence="medium", is_night=1)
    vid = add_visit(confidence="low", needs_review=1)
    for path in ("/false-triggers", "/", f"/visit/{vid}", "/about"):
        html = client.get(path).text
        pills = html.count('class="badge ')
        assert pills and len(re.findall(r'class="badge [^"]*" data-tip=', html)) == pills, path
    assert "Night: taken in infrared" in client.get("/false-triggers").text


def test_confirmed_pill_after_three_sure_visits(client, add_visit, monkeypatch):
    from birdbrain import species_info
    monkeypatch.setattr(species_info, "get", lambda conn, name, fetch=True: {"name": name, "common": name})
    for _ in range(2):
        add_visit(species="Bushtit", confidence="high")
    medium = add_visit(species="Bushtit", confidence="medium")
    assert "Not confirmed yet" in client.get("/species").text
    assert "Not confirmed yet" in client.get(f"/visit/{medium}").text
    add_visit(species="Bushtit", confidence="high")
    assert ">Confirmed<" in client.get("/species-info/Bushtit").text
    assert ">Confirmed<" in client.get(f"/visit/{medium}").text


def test_visit_page_arrows_step_through_false_triggers(client, add_visit):
    ft_old = add_visit(captured_at="2026-10-24T09:00:00", visitor="none", species=None)
    add_visit(captured_at="2026-10-24T10:00:00")
    ft_new = add_visit(captured_at="2026-10-24T11:00:00", visitor="none", species=None)
    assert f'href="/visit/{ft_new}?ft=1"' in client.get("/false-triggers").text
    html = client.get(f"/visit/{ft_new}").text
    assert f'href="/visit/{ft_old}?ft=1" id="nav-next"' in html and 'id="nav-prev"' not in html


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
    assert "<kbd>" not in client.get(f"/visit/{mid}").text


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
    client.post(f"/visit/{vid}/delete")
    assert images.burst_paths(image_rel) == []


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


def test_decision_steps_handles_refusals_and_old_or_broken_records():
    refused = json.dumps({"calls": {"haiku": {"answer": None}, "sonnet": {"frames": 3, "answer": None}}})
    steps = webapp.decision_steps(refused)["steps"]
    assert [s["refused"] for s in steps] == [True, True]
    assert webapp.decision_steps(None)["steps"] == []
    assert webapp.decision_steps('{"answer": {}}')["steps"] == []   # dev-seed rows have no trail
    assert webapp.decision_steps("not json")["steps"] == []


def test_false_triggers_page_shows_storage_and_cleans_up(client, stored_visit, conn):
    vid = stored_visit(visitor="none", species=None)
    stored_visit(visitor="none", species=None)  # not checked yet: stays
    stored_visit(visitor="none", species=None, feedback="rock tip")  # a lesson: survives clean-up
    html = client.get("/false-triggers").text
    assert "False-trigger photos:" in html and "of 500 MB" in html
    assert "1 photo you corrected or wrote a note on" in html
    assert 'class="storage-delete"' not in html  # nothing confirmed yet, so no Delete
    db.confirm_visits(conn, [vid])
    html = client.get("/false-triggers").text
    assert ">Delete</button>" in html and "1 photo you confirmed empty" in html
    r = client.post("/false-triggers/clean", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/false-triggers?cleared=1"
    html = client.get("/false-triggers?cleared=1").text
    assert "Deleted 1 photo. The Pi keeps what it learned from them." in html
    assert "without photos" not in html and f'href="/visit/{vid}?ft=1"' not in html
    assert db.count_visits(conn, false_triggers=True) == 3  # the record stays
    page = client.get(f"/visit/{vid}").text
    assert "Photo deleted to save space" in page and "/images/" + db.get_visit(conn, vid)["image_path"] not in page


def test_settings_page_changes_the_false_trigger_limit_and_photo_size(client, conn):
    html = client.get("/settings").text
    assert 'id="storage"' in html and 'value="500" checked' in html and 'value="medium" checked' in html
    assert "No limit" in html and "2 GB" in html and "Change the limit or photo size" not in html
    r = client.post("/settings/storage", data={"limit_mb": "250", "photo": "small"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/settings?pruned=0#storage"
    html = client.get("/settings?pruned=0").text
    assert 'value="250" checked' in html and 'value="small" checked' in html and "Saved." in html
    assert "of 250 MB" in client.get("/false-triggers").text
    assert 'href="/settings#storage"' in client.get("/false-triggers").text
    assert client.post("/settings/storage", data={"limit_mb": "250", "photo": "huge"}).status_code == 400


def test_settings_sections_collapse_and_show_a_status_when_closed(client, conn):
    html = client.get("/settings").text
    for section in ("local", "storage", "alerts"):
        assert f'<details class="settings-card" id="{section}" open>' in html
    assert ">Animal check shadow, species guess gate<" in html and ">Off<" in html
    notes = client.get("/notes").text
    for section in ("camera", "lessons"):
        assert f'<details class="settings-card" id="{section}" open>' in notes
    assert '<details class="settings-card prompt-preview" id="prompt">' in notes and ">No location set<" in notes
    alert.save_settings(conn, topic="t", new_species=True, mammals=True, quiet_start="", quiet_end="")
    db.save_settings(conn, location="Vancouver, WA")
    assert ">On: new species, raccoons and possums, out of season<" in client.get("/settings").text
    assert ">Vancouver, WA<" in client.get("/notes").text


def test_a_limit_from_env_that_is_not_a_button_still_shows_selected(client, monkeypatch):
    monkeypatch.setattr(config, "FALSE_TRIGGER_MAX_MB", 750.0)
    assert 'value="750" checked' in client.get("/settings").text


@pytest.mark.parametrize("value, text", [(0, "0.0 MB"), (420_500_000, "420 MB"), (1_500_000_000, "1.5 GB")])
def test_mb_formatting(value, text):
    assert webapp.mb(value) == text


def test_visit_page_shows_the_local_detector_step(client, add_visit):
    gated = json.dumps({"calls": {"detector": {"animal": False, "best": None, "seconds": 0.41}, "skipped_claude": True}})
    vid = add_visit(visitor="none", species=None, model="yolox-tiny", raw_json=gated)
    html = client.get(f"/visit/{vid}").text
    assert "Local detector" in html and "0.41 s" in html and "Claude wasn't asked" in html
    shadow = json.dumps({"calls": {"detector": {"animal": True, "best": {"label": "bird", "score": 0.62}, "seconds": 0.4},
                                   "haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}}}})
    html = client.get(f"/visit/{add_visit(species='Bushtit', raw_json=shadow)}").text
    assert "bird (62%)" in html and "Bushtit" in html and "Claude wasn't asked" not in html


def test_home_shows_visitor_count_as_text_and_stats_has_the_rest(client, add_visit, monkeypatch):
    raw = json.dumps({"calls": {"haiku": {"input_tokens": 2000, "output_tokens": 200}, "presence": {"empty": True}}})
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None, raw_json=raw)
    html = client.get("/").text
    assert 'class="tally-box tally-head"' in html and '<details class="tally-box' not in html
    assert "Sent to Claude" not in html and "Kept on the Pi" not in html
    assert html.index("Visitors today") < html.index("More filters")
    stats = client.get("/stats").text
    rows = ["Visitor", "False alarms", "Kept on the Pi", "Sent to Claude"]
    start = stats.index("Today's motion")
    assert [stats.index(r, start) for r in rows] == sorted(stats.index(r, start) for r in rows)
    assert "gate mode would have kept 1 on the Pi" in stats
    assert "50% of today's 2 motion events were a real animal" in stats
    monkeypatch.setattr(config, "PRESENCE", "gate")
    assert "gate mode would have kept" not in client.get("/stats").text
    monkeypatch.setattr(config, "PRESENCE", "off")
    assert "Kept on the Pi" not in client.get("/stats").text


@pytest.mark.parametrize("value, text", [(0, "$0"), (0.0004, "under $0.001"), (0.031, "$0.031"), (0.42, "$0.42")])
def test_usd_formatting(value, text):
    assert webapp.usd(value) == text


def test_visit_without_a_trail_hides_the_section(client, add_visit):
    assert "How it was decided" not in client.get(f"/visit/{add_visit()}").text


def test_claudes_notes_saves_the_description_and_shows_what_claude_is_told(client, conn):
    page = client.get("/notes").text
    assert "Your yard" in page and "Ripples and reflections right after a refill" in page  # example placeholder
    assert "Your yard" not in client.get("/settings").text  # moved off Settings
    r = client.post("/notes", data={"location": "Vancouver, Washington, USA", "scene_notes": "Rocks mid-water.",
                                       "ignore_notes": "Branch on the right sways."}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/notes?saved=1#camera"
    assert db.get_settings(conn)["ignore_notes"] == "Branch on the right sways."
    page = client.get("/notes?saved=1").text
    assert "Saved." in page and "The camera is in Vancouver, Washington, USA." in page  # the prompt preview
    assert 'href="/notes" aria-current="page"' in page  # nav link is highlighted
    assert 'title="Settings" aria-current="page"' in client.get("/settings").text  # gear is highlighted
    r = client.get("/lessons", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == "/notes#lessons"


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


def test_images_are_served(client, stored_visit, conn):
    row = db.get_visit(conn, stored_visit())
    r = client.get(f"/images/{row['thumb_path']}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_species_pages(client, add_visit):
    add_visit(species="Bushtit")
    assert "Bushtit" in client.get("/species").text
    assert "Bushtit" in client.get("/species/Bushtit").text
    assert client.get("/species/Nope").status_code == 404


@pytest.mark.parametrize("path", ["/stats", "/review", "/false-triggers", "/species?sort=name", "/species?sort=bogus"])
def test_pages_render(client, add_visit, path):
    add_visit(needs_review=1)
    assert client.get(path).status_code == 200


def test_latest_visit_api_ignores_false_triggers(client, add_visit):
    assert client.get("/api/latest-visit").json() == {"id": 0, "label": None, "captured_at": None, "any_id": 0}
    bird = add_visit(species="Bushtit")
    empty = add_visit(visitor="none", species=None)
    got = client.get("/api/latest-visit").json()
    assert (got["id"], got["label"], got["any_id"]) == (bird, "Bushtit", empty)


def test_home_and_visit_pages_pick_up_new_visitors_without_a_reload(client, add_visit):
    visit = add_visit(species="Bushtit")
    assert 'addEventListener("birdbrain:visits"' in client.get("/").text
    html = client.get(f"/visit/{visit}").text
    assert 'addEventListener("birdbrain:visits"' in html and "e.detail.real" in html
    assert "e.detail.any" in client.get(f"/visit/{add_visit(visitor='none', species=None)}").text


@pytest.mark.parametrize("path", ["/", "/stats", "/species", "/review", "/false-triggers"])
def test_every_page_has_the_theme_toggle_and_applies_the_saved_theme_before_paint(client, path):
    html = client.get(path).text
    assert 'id="theme-toggle"' in html and "/static/theme.js" in html
    # The saved-theme script must run before the stylesheet loads, or the page flashes the wrong theme.
    assert html.index('localStorage.getItem("theme")') < html.index("/static/style.css")


def test_favicon_and_static_assets_are_versioned(client):
    assert client.get("/favicon.ico").headers["content-type"] == "image/x-icon"
    html = client.get("/").text
    assert f"style.css?v={webapp.templates.env.globals['asset_version']}" in html


@pytest.fixture
def live(monkeypatch):
    """Fake camera grabs: each call pops the next scripted result."""
    state = {"jpeg": None, "at": 0.0, "tried": 0.0}
    monkeypatch.setattr(webapp, "_live", state)
    script: list[bytes] = []
    grabs = []

    async def grab():
        grabs.append(1)
        return script.pop(0) if script else b""

    monkeypatch.setattr(webapp, "grab_live_frame", grab)
    return state, script, grabs


def test_live_snapshot_is_shared_within_the_refresh_window(client, live):
    state, script, grabs = live
    script.append(b"frame-1")
    assert client.get("/live.jpg").content == b"frame-1"
    assert client.get("/live.jpg").content == b"frame-1"  # served from cache
    assert len(grabs) == 1


def test_live_reports_offline_instead_of_serving_a_stale_frame(client, live):
    state, script, grabs = live
    state.update(jpeg=b"old-frame", at=time.time() - webapp.LIVE_STALE - 1, tried=0.0)
    r = client.get("/live.jpg")  # the camera grab fails
    assert r.status_code == 503 and len(grabs) == 1


def test_live_with_no_frame_ever_is_503(client, live):
    assert client.get("/live.jpg").status_code == 503


@pytest.fixture
def frame_file(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "live_file", lambda: tmp_path / "live.jpg")
    return tmp_path / "live.jpg"


def test_fresh_frames_are_saved_and_report_their_age(client, live, frame_file):
    state, script, grabs = live
    script.append(b"fresh")
    r = client.get("/live.jpg")
    assert r.content == b"fresh" and float(r.headers["X-Frame-Age"]) < 1
    assert frame_file.read_bytes() == b"fresh"


def test_latest_frame_paints_instantly_without_touching_the_camera(client, live, frame_file):
    state, script, grabs = live
    state.update(jpeg=b"cached", at=time.time() - 600, tried=time.time())
    r = client.get("/live/latest.jpg")
    assert r.content == b"cached" and float(r.headers["X-Frame-Age"]) >= 600  # stale frames still shown, labeled
    assert client.head("/live/latest.jpg").status_code == 200
    assert grabs == []


def test_latest_frame_survives_a_restart_via_disk(client, live, frame_file):
    frame_file.write_bytes(b"from-before-restart")
    assert client.get("/live/latest.jpg").content == b"from-before-restart"


def test_latest_frame_404_when_there_has_never_been_one(client, live, frame_file):
    assert client.get("/live/latest.jpg").status_code == 404


def test_settings_shows_the_local_checks_as_one_flow(client, conn, monkeypatch):
    monkeypatch.setattr(config, "PRESENCE", "shadow")
    html = client.get("/settings").text
    assert 'id="local"' in html and 'id="detector"' not in html and 'value="shadow" checked' in html
    steps = ["Anything there?", "What species?", "Ask Claude"]
    assert [html.index(s) for s in steps] == sorted(html.index(s) for s in steps)
    assert "kept on the Pi" in html and "sent to Claude" in html
    r = client.post("/settings/presence", data={"mode": "gate"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("#local")
    assert client.post("/settings/detector", data={"mode": "gate"}).status_code in (404, 405)


@pytest.fixture
def calibrated(monkeypatch):
    monkeypatch.setattr(config, "MOTION_RING", (0.5, 0.5, 0.25, 0.3, 0.2, 0.22, 0.48))
    monkeypatch.setattr(config, "MOTION_SPOTS", ((0.5, 0.45, 0.04, 0.04),))
    monkeypatch.setattr(config, "CROP_ROI", (0.2, 0.1, 0.6, 0.8))
    monkeypatch.setattr(config, "MOTION_MIN_AREA", 0.02)


def test_about_page_explains_the_pipeline_with_live_values(client, add_visit, conn, calibrated):
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    db.set_value(conn, "presence_mode", "gate")
    html = client.get("/about").text
    assert "How Birdbrain works" in html and 'href="/about"' in html
    assert f'href="{config.REPO_URL}"' in html and "Source code on GitHub" in html
    assert "2%</strong> of the watched area" in html                            # MOTION_MIN_AREA, live
    assert f"{config.BURST_FRAMES} full-size photos" in html
    assert html.count('class="z-rim"') == 1 and html.count('class="z-spot"') == 1
    assert 'class="z-water"' in html and 'class="z-crop"' in html
    # each zone has the color it has on Settings > Zones: rim, water, rock, then crop (no ground here)
    colors = [webapp.SCENE_COLORS[i] for i in range(4)]
    for cls, color in zip(("z-rim", "z-water", "z-spot", "z-crop"), colors):
        assert f'class="{cls}" style="color: {color}"' in html
    assert "1 landing spot inside the water" in html
    assert re.search(r'mode on" role="listitem">\s*<strong>Gate</strong><span class="badge', html)  # mode marked
    assert "claude-haiku-5-5" in html and "claude-sonnet-5-5" in html
    assert "2</span> motion events checked" in html and "Example: a summer visitor" in html
    assert "—" not in html
    # diagrams first, then the numbers with visible titles, then the zones
    assert "How a visit gets spotted, identified and shared here" in html
    assert "The network path, from the bird to you" in html
    assert "How often visitors come" in html
    pos = [html.index(f'id="{i}"') for i in ("flow", "network", "numbers", "zones")]
    assert pos == sorted(pos) and '<a href="#numbers">Numbers</a>' in html


def test_about_page_without_calibration_or_data(client, monkeypatch):
    monkeypatch.setattr(config, "MOTION_RING", None)
    monkeypatch.setattr(config, "MOTION_ROI", None)
    monkeypatch.setattr(config, "CROP_ROI", None)
    html = client.get("/about").text
    assert "Whole frame" in html and "Claude sees the whole photo" in html
    assert 'class="z-water"' not in html and "Open water</h3>" not in html
    assert 'class="z-ground"' not in html and "Ground around the bath" not in html
    assert "motion events checked" not in html
    assert 'id="numbers"' not in html and 'href="#numbers"' not in html


def test_about_page_explains_the_ground_zone(client, calibrated, monkeypatch):
    monkeypatch.setattr(config, "GROUND_ZONE", ((0.1, 0.5), (0.9, 0.5), (0.9, 1), (0.1, 1)))
    html = client.get("/about").text
    assert 'class="z-ground"' in html and "Ground around the bath</h3>" in html
    assert f"own cap of {config.GROUND_MAX_CALLS_PER_DAY}" in html and "plus the ground around the bath" in html
    assert "0.6%</strong>" not in html and "about cat-sized (0.6%)" in html     # GROUND_MIN_BLOB_NIGHT, live


def test_zone_shapes_match_the_motion_masks(calibrated, monkeypatch):
    z = webapp.zone_shapes()
    assert z["rim"].startswith("M40.00,45.00 a40.00,27.00")                     # outer: cx .5, rx .25 of 160x90
    assert "M48.00,43.20 a32.00,19.80" in z["rim"]                              # inner ellipse uses inner_cy
    assert z["water"].count("M") == 2 and len(z["spots"]) == 1
    assert z["crop"] == {"x": 32.0, "y": 9.0, "w": 96.0, "h": 72.0}
    assert z["ground"] is None
    monkeypatch.setattr(config, "GROUND_ZONE", ((0.1, 0.5), (0.9, 0.5), (0.9, 1)))
    ground = webapp.zone_shapes()["ground"]
    assert ground.startswith("M16.00,45.00 L144.00,45.00 L144.00,90.00 Z")      # polygon in 160x90
    assert ground.endswith(webapp.ellipse_path(0.5, 0.5, 0.25, 0.3))            # minus the bath, like ground_mask


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
    assert f'class="vz-crop" style="color: {colors["crop"]}" x="32.0" y="9.0" width="96.0" height="72.0"' in html
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
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": then.to_dict()}))
    assert 'x="32.0" y="9.0" width="96.0" height="72.0"' in client.get(f"/visit/{vid}").text
    # An older visit (no zones stored) with no calibration history is drawn with today's, and says so.
    vid = add_visit(raw_json=json.dumps({"trigger": "rim"}), captured_at="2026-10-20T12:00:00")
    html = client.get(f"/visit/{vid}").text
    assert 'x="0.0" y="0.0" width="80.0" height="45.0"' in html and "today&#39;s zones" in html
    # With a history, the zones in use at the time: the last run before it, else the first run's before.
    moved = zones.Zones(crop=(0.5, 0.5, 0.25, 0.25))
    db.add_calibration(conn, ran_at="2026-10-21T08:00:00", reason="hand", status="moved", summary="",
                       before_json=json.dumps(then.to_dict()), after_json=json.dumps(moved.to_dict()))
    html = client.get(f"/visit/{vid}").text
    assert 'x="32.0" y="9.0" width="96.0" height="72.0"' in html and "calibration history" in html
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


def test_backoff_steps_stop_at_the_cap(monkeypatch):
    monkeypatch.setattr(config, "COOLDOWN_NONE", 20)
    monkeypatch.setattr(config, "COOLDOWN_NONE_MAX", 300)
    assert webapp.backoff_steps() == [20, 40, 80, 160, 300]


def test_season_example_prefers_a_real_card_with_a_season():
    year_round = {"local_counts": [50] * 12, "expected_months": list(range(1, 13))}
    summer = {"local_counts": [0, 0, 2, 10, 60, 90, 100, 80, 30, 5, 0, 0], "expected_months": [5, 6, 7, 8, 9]}
    ex = webapp.season_example({"American Robin": year_round, "Black-headed Grosbeak": summer})
    assert ex["name"] == "Black-headed Grosbeak" and ex["status"] == "Summer visitor"
    badges = {m["name"]: m["badge"] for m in ex["months"]}
    assert badges["Apr"] == "Early arrival" and badges["Oct"] == "Late departure" and badges["Jun"] is None
    assert webapp.season_example({})["name"] is None


def test_visit_page_says_which_zone_saw_the_motion(client, add_visit):
    vid = add_visit(species="Virginia Opossum", visitor="possum", raw_json=json.dumps({"trigger": "ground", "calls": {
        "haiku": {"answer": {"visitor": "possum", "species": "Virginia Opossum", "confidence": "high"}}}}))
    assert "Spotted moving on the ground." in client.get(f"/visit/{vid}").text
    assert webapp.decision_steps(json.dumps({"calls": {}}))["trigger"] is None   # --once captures have none


def test_latest_visitor_thumbnail_shows_claudes_circle(client, add_visit):
    add_visit(raw_json=json.dumps({"calls": {"circle": {"x": 0.45, "y": 0.65, "r": 0.05, "claude": True}}}))
    html = client.get("/").text
    assert "hero-photo" in html and html.count('class="marks thumb-marks"') == 2 and "0.45" in html  # big photo + grid card


def test_latest_visitor_takes_the_big_photo_and_live_view_moves_to_today(client, add_visit):
    html = client.get("/").text
    assert 'class="hero-latest"' not in html and html.index('id="live-box"') < html.index('class="today"')  # no visitor yet
    vid = add_visit(species="Bushtit", count=3)
    html = client.get("/").text
    hero = html[html.index('class="hero-latest"'):html.index('class="today"')]
    assert f'href="/visit/{vid}"' in hero and "Bushtit x3" in hero and "_t.jpg" not in hero  # full-size frame
    assert "hero-kicker-top" not in html and hero.index("hero-kicker") < hero.index("hero-name") < hero.index("hero-time")  # one chip, kicker first
    assert html.count('id="live-box"') == 1 and html.index('class="today"') < html.index('id="live-box"')


def test_false_triggers_come_in_pages(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "FT_PAGE", 2)
    for _ in range(3):
        add_visit(visitor="none", species=None)
    html = client.get("/false-triggers").text
    assert html.count('class="card"') == 2 and "/false-triggers/more?offset=2" in html
    more = client.get("/false-triggers/more?offset=2").text
    assert more.count('class="card"') == 1 and "?ft=1" in more and "load-more" not in more


def test_lessons_page_lists_lines_and_switches_them_off(client, add_visit, conn):
    vid = add_visit(corrected_visitor="none", feedback="me cleaning the camera")
    html = client.get("/notes").text
    assert "me cleaning the camera" in html and f'hx-post="/lessons/{vid}/off"' in html
    pills = html.count('class="badge ')
    assert pills and len(re.findall(r'class="badge [^"]*" data-tip=', html)) == pills
    assert client.post(f"/lessons/{vid}/off").headers["HX-Refresh"] == "true"
    assert db.recent_lessons(conn) == [] and "switched off" in client.get("/notes").text
    client.post(f"/lessons/{vid}/on")
    assert len(db.recent_lessons(conn)) == 1
    assert client.post(f"/lessons/{vid}/maybe").status_code == 404


def test_deleting_a_lesson_clears_the_visits_correction(client, add_visit, conn):
    vid = add_visit(corrected_visitor="none", feedback="me cleaning the camera")
    assert f'hx-post="/visit/{vid}/clear-correction"' in client.get("/notes").text
    client.post(f"/visit/{vid}/clear-correction")
    assert db.all_lessons(conn) == [] and db.get_visit(conn, vid)["feedback"] is None


def test_lesson_edit_changes_label_and_note_but_keeps_circles(client, add_visit, conn):
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    client.post(f"/visit/{vid}/correct", data={"visitor": "none", "feedback": "rock tip", "marks": '[{"x":0.5,"y":0.5,"r":0.1}]'})
    assert f'hx-post="/lessons/{vid}/edit"' in client.get("/notes").text
    client.post(f"/lessons/{vid}/edit", data={"visitor": "bird", "species": "Bushtit", "feedback": "a bushtit on the rocks"})
    line = db.recent_lessons(conn)[0]
    assert '"Bushtit"' in line and "a bushtit on the rocks" in line and "circle 1 at 50% across" in line
    assert client.post(f"/lessons/{vid}/edit", data={"visitor": "dragon"}).status_code == 400


def test_refine_with_claude_fills_the_boxes_without_saving(client, conn, monkeypatch):
    from birdbrain import refine

    def ask(location, scene_notes, target_notes, ignore_notes):
        assert scene_notes == "A big long rambling description of a birdbath in the yard"
        return refine.Refined(scene_notes="17.5 inch bath.", target_notes="Chickadees to ravens.",
                              ignore_notes="The rock.", changes="Moved sizes to look for."), \
            {"model": refine.MODEL, "input_tokens": 400, "output_tokens": 100}
    monkeypatch.setattr(refine, "ask", ask)
    r = client.post("/notes/refine", data={"location": "Vancouver",
                                           "scene_notes": "A big long rambling description of a birdbath in the yard"})
    assert r.status_code == 200 and 'id="yard-form"' in r.text and "Chickadees to ravens." in r.text
    assert "Moved sizes to look for." in r.text and "11 to 8 words" in r.text
    assert db.get_settings(conn)["scene_notes"] == ""  # only a suggestion until saved
    assert "Write something first." in client.post("/notes/refine", data={"location": "x"}).text
    assert "Refine with Claude" in client.get("/notes").text


def test_refine_keeps_the_owners_text_when_claudes_is_longer(client, monkeypatch):
    from birdbrain import refine
    monkeypatch.setattr(refine, "ask", lambda *a: (refine.Refined(
        scene_notes="Location: Vancouver. A round bath, 17.5 inches.", target_notes="", ignore_notes="", changes="x"),
        {"input_tokens": 1, "output_tokens": 1}))
    r = client.post("/notes/refine", data={"scene_notes": "Round bath, 17.5 inches."})
    assert "Already concise" in r.text and ">Round bath, 17.5 inches.</textarea>" in r.text


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


def test_false_trigger_cards_refresh_after_the_pop_up(client, add_visit, conn):
    keep = add_visit(visitor="none", species=None)
    real = add_visit(visitor="none", species=None, corrected_visitor="bird", corrected_species="Bushtit")
    db.confirm_visits(conn, [keep])
    r = client.get(f"/false-triggers/cards?ids={keep},{real},abc")
    assert f'data-id="{keep}"' in r.text and f'data-id="{real}"' not in r.text
    assert "Confirmed No Animal" in r.text
    assert r.headers["X-Unchecked"] == "0"


def test_visit_page_says_notes_are_for_claude_only(client, add_visit):
    html = client.get(f"/visit/{add_visit()}").text
    assert "Note for Claude" in html and "Notes are not used by the Pi's local animal check" in html
    assert "The answer you pick trains the Pi's local animal check" in html


def test_popups_use_the_site_dialog_not_the_browsers(client):
    # confirm()/alert()/prompt() can't be styled; questions go through static/dialog.js (birdbrain.ask,
    # hx-confirm, form[data-confirm]) and look like the rest of the site (the Pop-ups section of style.css).
    web = Path(webapp.__file__).parent
    native = re.compile(r"(?<![\w.])(?:window\.)?(?:confirm|alert|prompt)\(")
    for f in [*web.glob("templates/*.html"), *web.glob("static/*.js")]:
        assert not native.search(f.read_text()), f.name
    html = client.get("/").text
    assert html.index("/static/dialog.js") < html.index("/static/save-feedback.js")
    css = (web / "static/style.css").read_text()
    assert ".popup {" in css and ".viewer {" in css and ".popup-actions" in css


def test_species_list_names_open_the_info_card_and_card_has_jump_to_visits(client, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    assert 'data-species="Bushtit"' in client.get("/species").text
    from birdbrain import species_info
    monkeypatch.setattr(species_info, "get", lambda conn, name, fetch=True: {"name": name, "common": name, "status": "Year-round resident"})
    html = client.get("/species-info/Bushtit").text
    assert 'href="/species/Bushtit#visits"' in html and "Jump to visits" in html
    assert "All visits" not in html and 'class="sp-facts"' in html
    assert 'id="visits"' in client.get("/species/Bushtit").text
    assert re.findall(r'class="badge [^"]*"(?! data-tip)', html) == []


def test_home_photos_carry_a_zoom_button_but_other_pages_do_not(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "PAGE_SIZE", 1)
    first = add_visit(image_path="2026/11/01/a.jpg", thumb_path="2026/11/01/a_t.jpg", species="Bushtit")
    second = add_visit(image_path="2026/11/01/b.jpg", thumb_path="2026/11/01/b_t.jpg", species="Dark-eyed Junco")
    html = client.get("/").text
    hero = html[html.index('class="hero-latest"'):html.index('class="today"')]
    assert 'href="/visit/%d"' % second in hero and 'data-zoom-photo="/images/2026/11/01/b.jpg"' in hero
    assert hero.index("</a>") < hero.index("zoom-btn")  # a sibling of the link, not inside it
    card = html[html.index('class="card-zoom"'):]
    assert 'data-zoom-photo="/images/2026/11/01/b.jpg"' in card
    assert html.count('class="card-zoom"') == 1 and "live-zoom-btn" in html and 'id="home-viewer"' in html
    assert f'<a class="hero-goto" href="/visit/{second}">Go to visit</a>' in hero  # after the kicker, a real link
    assert hero.index("hero-kicker") < hero.index("hero-goto") < hero.index("hero-name")
    assert 'id="home-viewer-reset"' in html and "Reset view" in html  # zoom viewer
    # Load more cards get it too
    more = client.get("/visits/more?offset=1").text
    assert f'href="/visit/{first}"' in more and 'data-zoom-photo="/images/2026/11/01/a.jpg"' in more
    # species and false-trigger pages are unchanged
    assert "zoom-btn" not in client.get("/species/Bushtit").text
    add_visit(visitor="none", species=None)
    assert "zoom-btn" not in client.get("/false-triggers").text


def test_manifest_has_maskable_icon_and_base_path(client, monkeypatch):
    from birdbrain.web import app as webapp
    r = client.get("/manifest.webmanifest")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/manifest+json")
    m = r.json()
    assert m["start_url"] == "/" and m["scope"] == "/" and m["display"] == "standalone"
    assert any(i["purpose"] == "maskable" for i in m["icons"])
    assert 'rel="manifest"' in client.get("/").text
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    m = client.get("/manifest.webmanifest").json()
    assert m["start_url"] == "/birdbrain/" and m["scope"] == "/birdbrain/"
    assert all(i["src"].startswith("/birdbrain/static/") for i in m["icons"])
    assert 'href="/birdbrain/manifest.webmanifest' in client.get("/").text


def test_visits_archive_lists_real_visits_only(client, add_visit):
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    html = client.get("/visits").text
    assert html.count('class="card"') == 1 and "Bushtit" in html and "1 visit<" in html
    assert 'href="/visits" aria-current="page"' in html


def test_nav_has_visits_link(client):
    assert 'href="/visits"' in client.get("/").text


def test_visits_archive_load_more_and_filters(client, add_visit):
    for _ in range(30):
        add_visit(species="Bushtit")
    add_visit(visitor="raccoon", species=None)
    first = client.get("/visits").text
    assert first.count('class="card"') == 24 and "/visits/archive-more?offset=24" in first and "31 visits" in first
    more = client.get("/visits/archive-more?offset=24").text
    assert more.count('class="card"') == 7 and "load-more" not in more and "grid-break" not in more
    assert "1 visit matching" in client.get("/visits?kind=mammal").text


def test_visit_opened_from_species_page_keeps_the_species_scope(client, add_visit):
    old = add_visit(captured_at="2026-10-24T09:00:00", species="Bushtit")
    sparrow = add_visit(captured_at="2026-10-24T10:00:00", species="Song Sparrow")
    new = add_visit(captured_at="2026-10-24T11:00:00", species="Bushtit")
    assert f'href="/visit/{new}?species=Bushtit"' in client.get("/species/Bushtit").text
    html = client.get(f"/visit/{new}?species=Bushtit").text
    assert f'href="/visit/{old}?species=Bushtit" id="nav-next"' in html
    assert f'class="photo-arrow next" href="/visit/{old}?species=Bushtit"' in html
    assert "Browsing: Bushtit" in html
    plain = client.get(f"/visit/{new}").text
    assert "Browsing:" not in plain and "species=Bushtit" not in plain
    assert f'href="/visit/{sparrow}" id="nav-next"' in plain


def test_tour_script_and_rerun_buttons(client):
    assert "/static/tour.js" in client.get("/").text
    assert "data-tour-restart" in client.get("/about").text
    assert "data-tour-restart" in client.get("/settings").text
    assert "data-admin=\"1\"" in client.get("/").text  # no password: everyone is admin
    assert "tour.js" not in client.get("/?embed=1").text
