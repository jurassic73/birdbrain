"""Component tests of the web app through TestClient on a temp database: pages."""

import io
import json
import re
from datetime import timedelta
from pathlib import Path

import pytest
from PIL import Image

from helpers.web import FOUR, calibrated
from birdbrain import alert, config, db, images
from birdbrain.web import app as webapp


def test_every_tip_label_has_its_own_info_icon(client, stored_visit):
    """Hovering a heading's words shows its info icon's tip (tips.js): the icon must be a child of the
    .tip-label or its next element sibling, and the label must hold no data-tip of its own."""
    from html.parser import HTMLParser

    class Labels(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth, self.open, self.closed, self.found = 0, [], None, []  # open: (depth, index)

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            cls = a.get("class", "").split()
            if "info-tip" in cls and "data-tip" in a:
                if self.open:  # a child of the innermost open label
                    self.found[self.open[-1][1]] = True
                elif self.closed is not None:  # the first element after a label closed
                    self.found[self.closed] = True
            if "tip-label" in cls:
                assert "data-tip" not in a
                self.open.append((self.depth, len(self.found)))
                self.found.append(False)
            self.closed = None
            if tag not in ("input", "img", "br", "meta", "link"):
                self.depth += 1

        def handle_endtag(self, tag):
            self.depth -= 1
            self.closed = None
            if self.open and self.open[-1][0] == self.depth:
                i = self.open.pop()[1]
                if not self.found[i]:
                    self.closed = i

    vid = stored_visit(visitor="none", species=None, raw_json=json.dumps({"burst": {"count": 3, "best": 1, "fps": 4}}))
    with db.session() as conn:
        image_rel = db.get_visit(conn, vid)["image_path"]
    buf = io.BytesIO()
    Image.new("RGB", (640, 360)).save(buf, "JPEG")
    images.store_burst([buf.getvalue()] * 3, image_rel)
    for path, n in ((f"/visit/{vid}", 6), ("/false-triggers", 1)):
        p = Labels()
        p.feed(client.get(path).text)
        assert len(p.found) == n and all(p.found), (path, p.found)


def test_admins_can_always_draw_circles_without_a_button(client, add_visit):
    ft = add_visit(visitor="none", species=None)
    real = add_visit(captured_at="2026-10-24T10:00:00")
    for url in (f"/visit/{ft}?ft=1", f"/visit/{ft}?ft=1&todo=1&embed=1", f"/visit/{real}"):  # incl. the pop-up
        html = client.get(url).text
        assert "const editing = true;" in html
        assert 'id="draw-note"' in html and "Add circles: drag on the photo" in html
        assert 'id="mark-edit"' not in html  # the old Add circles / Done button is gone
        assert 'id="mark-clear"' not in html and "Clear circles" not in html  # and so is Clear circles
        # The overlay switches sit at the far right of the row above the photo, zones last.
        end = html.index('class="photo-tools-end"')
        assert end < html.index('id="circles-toggle"')
        assert 'id="zones-toggle"' not in html or html.index('id="circles-toggle"') < html.index('id="zones-toggle"')


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
    assert ">Pi animal check shadow, BioCLIP 2 species check gate<" in html and ">Off<" in html
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


def test_stats_today_card_is_a_funnel_that_adds_up(client, add_visit):
    html = client.get("/stats").text
    assert 'id="today"' in html and "No motion yet today." in html and 'class="funnel"' not in html
    add_visit(species="Bushtit", model="bioclip-2")
    add_visit(species="Dark-eyed Junco", model="claude-haiku-5-5")
    add_visit(species="Dark-eyed Junco", model="claude-haiku-5-5")
    add_visit(visitor="none", species=None, model="bioclip-presence")
    html = client.get("/stats").text
    card = html[html.index('id="today"'):html.index('class="charts"')]
    assert "Today</h2>" in card and ", so far" in card
    assert "Motion events" in card and "Pi animal check" in card and "Named by" in card and "Visitors" in card
    assert '<span class="n">4</span>' in card and "&minus;1</b>" in card and '<span class="n good">3</span>' in card
    assert "2 species" in card and "BioCLIP 2 &middot; free" in card
    assert '<span class="n">1</span><span class="t">BioCLIP 2' in card and '<span class="n">2</span><span class="t">Claude' in card
    assert "75%</b> of motion was a real animal" in card
    assert "Today's Visits" not in html and "Today's motion" not in html and "tally" not in html
    assert html.index('id="today"') < html.index("Visits per day") < html.index("Most frequent")


def test_stats_today_card_zero_states(client, add_visit):
    add_visit(visitor="none", species=None, model="bioclip-presence")  # motion, but nobody came
    card = client.get("/stats").text
    card = card[card.index('id="today"'):card.index('class="charts"')]
    assert "no visitors to name yet" in card and '<span class="n good">0</span>' in card
    assert "0%</b> of motion" in card and "per visitor" not in card


def test_stats_today_card_says_when_claude_checked_empties_and_gate_would_help(client, add_visit, monkeypatch):
    raw = json.dumps({"calls": {"haiku": {"input_tokens": 2000, "output_tokens": 200}, "presence": {"empty": True}}})
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None, raw_json=raw)
    stats = client.get("/stats").text
    assert "checked by Claude" in stats and "Claude was asked about 1 and found nothing" in stats
    assert "gate mode would have kept 1 on the Pi, saving about" in stats
    assert "50%</b> of motion was a real animal" in stats
    assert 'href="/false-triggers">empty</a>' in stats  # no password set: everyone is the admin
    monkeypatch.setattr(config, "PRESENCE", "gate")
    assert "gate mode would have kept" not in client.get("/stats").text
    monkeypatch.setattr(config, "PRESENCE", "off")
    off = client.get("/stats").text
    assert "Pi animal check" not in off[off.index('id="today"'):off.index('class="charts"')]


def test_stats_all_time_label(client, add_visit, conn):
    assert 'class="muted page-total" id="all-time">0 visits<' in client.get("/stats").text
    add_visit(species="Bushtit", captured_at=(webapp.clock.now() - timedelta(days=10)).isoformat(timespec="seconds"))
    add_visit(species="Bushtit")
    html = client.get("/stats").text
    label = html[html.index('id="all-time">'):html.index("</span>", html.index('id="all-time">'))]
    assert label.startswith('id="all-time">All time: 2 visits since ') and label.endswith("(~10 days ago)")


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


@pytest.mark.parametrize("path", ["/stats", "/review", "/false-triggers", "/species?sort=name", "/species?sort=bogus"])
def test_pages_render(client, add_visit, path):
    add_visit(needs_review=1)
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/", "/stats", "/species", "/review", "/false-triggers"])
def test_every_page_has_the_theme_toggle_and_applies_the_saved_theme_before_paint(client, path):
    html = client.get(path).text
    assert 'id="theme-toggle"' in html and "/static/theme.js" in html
    # The saved-theme script must run before the stylesheet loads, or the page flashes the wrong theme.
    assert html.index('localStorage.getItem("theme")') < html.index("/static/style.css")


def test_settings_shows_the_local_checks_as_one_flow(client, conn, monkeypatch):
    monkeypatch.setattr(config, "PRESENCE", "shadow")
    html = client.get("/settings").text
    assert 'id="local"' in html and 'id="detector"' not in html and 'value="shadow" checked' in html
    steps = ["(Anything there?)", "(What species?)", "3</span> Claude"]
    assert [html.index(s) for s in steps] == sorted(html.index(s) for s in steps)
    assert "kept on the Pi" in html and "sent to Claude" in html
    r = client.post("/settings/presence", data={"mode": "gate"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("#local")
    assert client.post("/settings/detector", data={"mode": "gate"}).status_code in (404, 405)


@pytest.mark.smoke
def test_about_page_explains_the_pipeline_with_live_values(client, add_visit, conn, calibrated):
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    db.set_value(conn, "presence_mode", "gate")
    html = client.get("/about").text
    assert "How Birdbrain works" in html and 'href="/about"' in html
    assert "Source code on GitHub" not in html
    assert "2%</strong> of the watched area" in html                            # MOTION_MIN_AREA, live
    assert f"{config.BURST_FRAMES} full-size photos" in html
    assert html.count('class="z-rim"') == 1 and html.count('class="z-spot"') == 1
    assert 'class="z-water"' in html and 'class="z-crop"' in html
    # each zone has the color it has on Settings > Zones: rim, water, rock, then crop (no ground here)
    colors = [webapp.SCENE_COLORS[i] for i in range(4)]
    for cls, color in zip(("z-rim", "z-water", "z-spot", "z-crop"), colors):
        assert f'class="{cls}" style="color: {color}"' in html
    assert "1 landing spot inside the water" in html
    assert re.search(r'The Pi animal check <span class="badge pi-only"[^>]*>gate</span>', html)  # mode marked, live
    assert "claude-haiku-5-5" in html and "claude-sonnet-5-5" in html
    assert "—" not in html
    # the 10,000-foot view: no numbers, seasons, cooldowns or JSON; diagrams first, then the zones and rules
    for gone in ("motion events checked", "How often visitors come", "Example: a summer visitor", "Cooldowns and rainy days",
                 '"visitor": "bird"', 'id="numbers"', 'id="seasons"', 'href="#numbers"', 'href="#seasons"'):
        assert gone not in html, gone
    assert "How a visit gets spotted, identified and shared here" in html
    assert "The network path, from the bird to you" in html
    pos = [html.index(f'id="{i}"') for i in ("flow", "handled", "network", "zones", "rules", "photos", "learning", "sources",
                                             "storage", "stack")]
    assert pos == sorted(pos) and '<a href="#photos">Photos to Claude</a>' in html
    # which parts learn and which don't, one row each with its pill
    learn = html[html.index('id="learning"'):html.index('id="sources"')]
    assert learn.count("<tr>") == 4 and all(n in learn for n in FOUR)
    assert ">can&#39;t be trained<" in learn or ">can't be trained<" in learn
    assert "in-context learning, not fine-tuning" in learn and f"{db.CONFIRMED_MIN} sure visits" in learn


@pytest.mark.smoke
def test_about_page_without_calibration_or_data(client, monkeypatch):
    monkeypatch.setattr(config, "MOTION_RING", None)
    monkeypatch.setattr(config, "MOTION_ROI", None)
    monkeypatch.setattr(config, "CROP_ROI", None)
    html = client.get("/about").text
    assert "Whole frame" in html and "Claude sees the whole photo" in html
    assert 'class="z-water"' not in html and "Open water</h3>" not in html
    assert 'class="z-ground"' not in html and "Ground around the bath" not in html


def test_about_page_explains_the_ground_zone(client, calibrated, monkeypatch):
    monkeypatch.setattr(config, "GROUND_ZONE", ((0.1, 0.5), (0.9, 0.5), (0.9, 1), (0.1, 1)))
    html = client.get("/about").text
    assert 'class="z-ground"' in html and "Ground around the bath</h3>" in html
    assert "plus the ground around the bath" in html
    assert "0.6%</strong>" not in html and "about cat-sized (0.6%)" in html     # GROUND_MIN_BLOB_NIGHT, live


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


def test_about_has_the_four_steps_as_numbered_cards_with_live_numbers(client):
    from birdbrain import classifier, db, lessons
    html = client.get("/about").text
    start = html.index('id="handled"')
    section = html[start:html.index("</section>", start)]
    assert "How a photo is handled" in section and len(re.findall(r'class="hs-step"', section)) == 4
    for name in FOUR:
        assert f"What the {name} does" in section and f"How the {name} is trained" in section
    assert "What you control" in section
    assert f"{config.TRIGGER_FRAMES} frames in a row" in section and f"{classifier.GATE_MIN_SCORE_BOWL:.0%}" in section
    assert f"{classifier.GATE_MIN_SCORE_TIGHT:.0%}" in section
    assert f"at least {db.CONFIRMED_MIN} earlier visits" in section and f"newest {db.LESSON_LIMIT} lessons" in section
    assert f"After every {lessons.AUTO_EVERY} new lessons" in section
    assert len(re.findall(r"<summary>More</summary>", section)) == 4  # training and controls fold under More
    assert "Anything there?" not in html  # the page says Pi animal check, never the Settings sub-label


def test_no_page_calls_the_pi_animal_check_by_an_old_name(client, add_visit):
    vid = add_visit(visitor="none", species=None)
    for path in ("/false-triggers", "/about", "/settings", "/stats", f"/visit/{vid}"):
        html = client.get(path).text
        for old in ("Local animal check", "local animal check", "Local classifier", "Local checks", "the Pi's animal check",
                    "The Pi's animal check"):
            assert old not in html, (path, old)


@pytest.mark.parametrize("path", ["/", "/species", "/stats", "/review", "/false-triggers", "/notes"])
def test_top_level_pages_turn_on_the_phone_swipe(client, path):
    html = client.get(path).text
    assert re.search(r'<body[^>]*data-swipe="nav"', html)
    assert f"/static/swipe.js?v={webapp.templates.env.globals['asset_version']}" in html


@pytest.mark.parametrize("path", ["/about", "/settings", "/species/Junco", "/visit/1"])
def test_other_pages_do_not_swipe_between_tabs(client, add_visit, path):
    add_visit()
    html = client.get(path).text
    assert "data-swipe=" not in html


def test_theme_color_html_never_gives_a_dark_device_the_light_color(client, add_visit):
    """The Android bar went light on a dark page after opening a visit: the HTML's only theme-color was the light
    fallback. Every page now carries a device-scheme pair (right even before or without the script), and the
    script keeps exactly one."""
    import re
    vid = add_visit(species="Bushtit")
    for url in ("/", "/?kind=bird", f"/visit/{vid}", f"/visit/{vid}?embed=1", "/species", "/stats"):
        html = client.get(url).text
        metas = re.findall(r'<meta name="theme-color"[^>]*>', html)
        assert metas == ['<meta name="theme-color" id="theme-color" content="#f6f5f0" '
                         'media="(prefers-color-scheme: light)">',
                         '<meta name="theme-color" content="#141815" media="(prefers-color-scheme: dark)">'], url
        assert html.index("window.bbTheme.apply();") < html.index('<link rel="stylesheet"'), url


def test_theme_color_is_reapplied_however_a_page_is_shown():
    """Prerendered visit pages (home's speculation rules), back/forward cache restores, a returning tab, the end of
    full screen, another tab's toggle and htmx swaps all re-apply the theme and bar color; the shown cases also
    nudge Chrome so it hears the colors again."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "birdbrain/web"
    js = (root / "static/theme.js").read_text()
    for hook in ('addEventListener("pageshow", refresh)', '"prerenderingchange", refresh', '"visibilitychange"',
                 '"fullscreenchange"', 'addEventListener("storage"', '"htmx:afterSettle"'):
        assert hook in js, hook
    assert "window.bbTheme.nudge()" in js and "window.bbTheme.apply(read)" in js
    base = (root / "templates/base.html").read_text()
    # one meta kept, the rest removed; the nudge restores the real color and drops its inline background
    assert 'querySelectorAll(\'meta[name="theme-color"]\').forEach(x => { if (x !== m) x.remove(); })' in base
    assert 'root.style.removeProperty("background-color")' in base and "document.prerendering" in base


def test_stats_share_chart_card_only_once_bioclip_named_a_visitor(client, add_visit):
    add_visit(species="Dark-eyed Junco", model="claude-haiku-5-5")
    assert 'id="id-share"' not in client.get("/stats").text
    add_visit(species="Bushtit", model="bioclip-2")
    html = client.get("/stats").text
    assert 'id="id-share"' in html and "Share of IDs by BioCLIP 2, per day" in html
    assert 'shareChart("idshare", [{' in html and '"bioclip":1' in html.replace(" ", "")
    assert html.index('id="today"') < html.index('id="id-share"')
