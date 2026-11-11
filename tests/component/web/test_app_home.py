"""Component tests of the web app through TestClient on a temp database: home."""

import pytest
import json
import re
import time
from datetime import timedelta
from pathlib import Path

from PIL import Image

from helpers.clock import FROZEN_NOW
from helpers.web import _card, card_ids, _keys_dialog, more_link
from birdbrain import db
from birdbrain.web import app as webapp


def test_home_right_arrow_opens_the_latest_visit(client, add_visit):
    vid = add_visit(species="Dark-eyed Junco")
    html = client.get("/").text
    assert f'class="hero-link" href="/visit/{vid}"' in html
    assert 'e.key !== "ArrowRight"' in html and 'querySelector(".hero-link")' in html
    assert "ArrowLeft" not in html.split('e.key !== "ArrowRight"')[1].split("</script>")[0]


def test_visit_escape_goes_home_only_after_arrow_trail(client, add_visit):
    vid = add_visit(species="Dark-eyed Junco")
    # The trail rides in the URL fragment, not sessionStorage (a prerendered visit page can see a stale copy).
    home = client.get("/").text
    assert '"#from-home"' in home and "sessionStorage.setItem" not in home.split('e.key !== "ArrowRight"')[1].split("</script>")[0]
    # The latest-visitor photo isn't prerendered: a prerender swapped in by the arrow key may lack focus and lose the first Esc.
    rules = json.loads(re.search(r'<script type="speculationrules">(.*?)</script>', home, re.S).group(1))
    assert {"not": {"selector_matches": ".hero-link"}} in rules["prerender"][0]["where"]["and"]
    html = client.get(f"/visit/{vid}").text
    assert 'location.hash !== "#from-home"' in html and "history.replaceState" in html
    assert 'addEventListener("hashchange", readTrail)' in html
    assert 'stepVisit("nav-next")' in html and 'a.href.split("#")[0] + "#from-home"' in html  # arrows pass it on
    assert 'location.href = (window.BB_BASE || "") + "/"' in html and "bb_arrow_trail" not in html
    assert 'e.key !== "ArrowUp"' in html and "if (scrollY > 0) return;" in html  # Up at the top goes home, any visit
    embed = client.get(f"/visit/{vid}?embed=1").text
    assert 'e.key !== "Escape"' not in embed and 'e.key !== "ArrowUp"' not in embed


def test_home_empty_state_hides_filters(client):
    html = client.get("/").text
    assert "No visitors yet." in html and 'class="filters-toggle"' not in html and 'id="more-filters"' not in html


@pytest.mark.smoke
def test_home_lists_visits_with_chips_and_filters_toggle(client, add_visit):
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    html = client.get("/").text
    assert "Bushtit" in html and "tally-box" not in html and "Mammals" in html
    # More filters is a toggle on the tally row, above the chips; its boxes start hidden. The Today panel has no Last motion line any more.
    assert 'class="filters-toggle"' in html and 'aria-expanded="false"' in html and 'id="more-filters"' in html
    assert html.index('class="filters-toggle"') < html.index('class="chips"') < html.index('id="more-filters"')
    assert "Last motion" not in html


def test_home_more_filters_open_when_a_species_is_picked(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/?species=Bushtit").text
    assert 'aria-expanded="true"' in html
    form = html[html.index('id="more-filters"'):]
    assert form[:form.index(">")].count("hidden") == 0


def test_home_lists_todays_visitors_with_counts(client, add_visit):
    assert "None yet today." in client.get("/").text
    add_visit(species="Bushtit")
    add_visit(species="Bushtit")
    add_visit(visitor="squirrel", species=None)
    add_visit(visitor="none", species=None)
    critters = client.get("/").text.split('class="critters"')[1].split("</ul>")[0]
    assert critters.index("Bushtit") < critters.index("Squirrel")
    assert ">2</a>" in critters and ">1</a>" in critters and "None" not in critters


def test_today_list_shows_the_seasonal_status_as_a_pill_after_the_info_icon(client, conn, add_visit):
    from birdbrain import species_info
    add_visit(species="Dark-eyed Junco")
    _card(conn, "Dark-eyed Junco", status="Year-round resident")
    species_info.forget_cached_cards()
    row = client.get("/").text.split('class="critters"')[1].split("</li>")[0]
    icon = row.index('class="info-tip card-tip"')
    pill = row.index('class="badge sp-status"')
    assert icon < pill and 'data-tip="Year-round resident: reported here' in row[pill:]
    assert ">Year-round</span>" in row[pill:] and 'class="critter-status"' not in row


def test_home_facts_box_shows_the_latest_species_facts_with_a_link(client, conn, add_visit):
    add_visit(species="Dark-eyed Junco")
    add_visit(species="Bushtit", captured_at=(FROZEN_NOW - timedelta(minutes=5)).isoformat())
    pool = ["Juncos flash white tail feathers.", "Flocks keep a pecking order.", "A third.", "A fourth."]
    _card(conn, "Dark-eyed Junco", facts=pool)
    _card(conn, "Bushtit", facts=["Bushtits travel in noisy flocks."])
    box = client.get("/").text.split('class="today-facts"')[1].split("</section>")[0]
    assert '<a href="/species/Dark-eyed%20Junco">Dark-eyed Junco</a>' in box
    shown = re.findall(r"<li>(.*?)</li>", box)
    assert len(shown) == 3 and set(shown) < set(pool)  # 3 of the pool, only the latest visitor's species
    assert "Bushtits travel" not in box
    assert client.get("/").text.count('class="today-facts"') == 1


def test_home_facts_box_rotates_through_the_pool(client, conn, add_visit):
    """A card holds a pool of facts; the three on show move on with every new visit (and every few minutes)."""
    from birdbrain import species_info

    def shown():
        box = client.get("/").text.split('class="today-facts"')[1].split("</section>")[0]
        return re.findall(r"<li>(.*?)</li>", box)

    pool = [f"Junco fact number {i}." for i in range(9)]
    _card(conn, "Dark-eyed Junco", facts=pool)
    first_id = add_visit(species="Dark-eyed Junco", captured_at=(FROZEN_NOW - timedelta(minutes=9)).isoformat())
    first = shown()
    assert first == species_info.rotate(pool, species_info.fact_turn(extra=first_id), 3)
    assert shown() == first  # a reload inside the window shows the same three
    add_visit(species="Dark-eyed Junco", captured_at=(FROZEN_NOW - timedelta(minutes=2)).isoformat())
    second = shown()
    assert len(second) == 3 and not set(second) & set(first)  # the next visit brings three new ones


def test_home_facts_box_kicker_says_latest_or_last_before_today(client, conn, add_visit):
    add_visit(species="Dark-eyed Junco")
    _card(conn, "Dark-eyed Junco", facts=["Juncos flash white tail feathers."])
    box = client.get("/").text.split('class="today-facts"')[1].split("</section>")[0]
    assert '<span class="facts-kicker">Latest visitor</span>' in box
    assert box.index("facts-kicker") < box.index("<h3>")
    conn.execute("UPDATE visits SET captured_at = ?", ((FROZEN_NOW - timedelta(days=2)).isoformat(),))
    conn.commit()
    box = client.get("/").text.split('class="today-facts"')[1].split("</section>")[0]
    assert '<span class="facts-kicker">Last visitor, before today</span>' in box


def test_home_facts_box_is_left_out_without_a_card_or_facts(client, conn, add_visit):
    assert 'class="today-facts"' not in client.get("/").text  # no visitors at all
    add_visit(species="Dark-eyed Junco")
    assert 'class="today-facts"' not in client.get("/").text  # no card cached
    _card(conn, "Dark-eyed Junco", facts=[])
    assert 'class="today-facts"' not in client.get("/").text  # card without facts
    _card(conn, "Dark-eyed Junco", description="A sparrow.")
    assert 'class="today-facts"' not in client.get("/").text  # old card from before facts existed
    _card(conn, "Dark-eyed Junco", facts=["", "  ", 5])
    assert 'class="today-facts"' not in client.get("/").text  # blanks don't count


def test_home_facts_box_handles_dict_and_plain_string_facts(client, conn, add_visit):
    add_visit(species="Dark-eyed Junco")
    _card(conn, "Dark-eyed Junco", facts=[
        {"lead": "Tail flash", "text": "Juncos flash white tail feathers."},
        {"lead": "", "text": "   "},                      # blank text is skipped
        {"lead": "Pecking order", "text": ""},
        "An old plain-string fact.",                      # cards saved before the {lead, text} change
        {"lead": "Ground feeders", "text": "They hop along the ground."},
    ])
    box = client.get("/").text.split('class="today-facts"')[1].split("</section>")[0]
    assert "Juncos flash white tail feathers." in box and "An old plain-string fact." in box
    assert "They hop along the ground." in box and box.count("<li>") == 3  # the limit of three: the rotation test
    assert "Tail flash" not in box and "Ground feeders" not in box  # the lead isn't shown, only the sentence
    assert box.count("<li>") == 3


def test_home_facts_box_skips_labels_without_a_card(client, conn, add_visit):
    add_visit(visitor="squirrel", species=None)
    _card(conn, "Squirrel", facts=["Squirrels bury thousands of nuts."])
    assert "Squirrels bury" in client.get("/").text
    _card(conn, "Other", facts=["Never shown."])
    add_visit(visitor="other", species="Other", captured_at=(FROZEN_NOW + timedelta(minutes=1)).isoformat())
    assert "Never shown." not in client.get("/").text


def test_home_facts_box_reads_the_cache_only(client, conn, add_visit, monkeypatch):
    from birdbrain import species_info
    def boom(*a, **k):
        raise AssertionError("the home page must not build or fetch a card")
    monkeypatch.setattr(species_info, "build", boom)
    monkeypatch.setattr(species_info, "_rebuild", boom)
    add_visit(species="Dark-eyed Junco")
    assert 'class="today-facts"' not in client.get("/").text  # nothing cached, nothing built
    # A card older than the monthly refresh is still shown (visitors never trigger the rebuild).
    db.save_species_info(conn, "Dark-eyed Junco", {"name": "Dark-eyed Junco", "facts": ["Old but fine."]}, "2020-01-01T00:00:00")
    assert "Old but fine." in client.get("/").text


def test_scoreboard_periods(client, add_visit):
    add_visit(species="Bushtit", captured_at=(FROZEN_NOW - timedelta(days=3)).replace(microsecond=0).isoformat())
    assert "None yet today." in client.get("/scoreboard?range=today").text
    assert "None yesterday." in client.get("/scoreboard?range=yesterday").text
    week = client.get("/scoreboard?range=week").text
    assert '<strong class="score-total">1</strong> Visitor this week' in week and "Bushtit" in week and 'class="score-range on"' in week
    assert '<strong class="score-total">0</strong> Visitors today' in client.get("/scoreboard?range=bogus").text
    assert "Bushtit" in client.get("/scoreboard?range=all").text
    # The This week chip shows the week on the scoreboard too.
    href = next(f for n, f in webapp.quick_filters() if n == "This week")
    assert "Visitor this week" in client.get("/?" + webapp.query_string(href)).text


def test_home_filter_chip_is_marked_active(client, add_visit):
    add_visit(visitor="raccoon", species=None)
    html = client.get("/?kind=mammal").text
    assert 'aria-current="page">Mammals<' in html and "matching</h2>" in html


def test_multiple_chip_follows_mammals_and_filters(client, add_visit):
    add_visit(species="Bushtit", also='[{"visitor": "bird", "species": "Song Sparrow", "count": 1}]')
    add_visit(species="Dark-eyed Junco", count=2)
    add_visit(species="Dark-eyed Junco", count=1)
    html = client.get("/?kind=multiple").text
    assert 'aria-current="page">Multiple<' in html and "2 matching</h2>" in html
    assert html.index(">Mammals<") < html.index(">Multiple<") < html.index(">Night<")
    assert '<option value="multiple" selected>' in html


def test_more_filters_form_is_fetched_in_place_and_empty_fields_are_optional(client, add_visit):
    """Apply and Clear swap the page in place (home.html builds /?kind=bird without the empty boxes), so the
    URL it fetches must give the same page as the form's own GET with empty fields, and the script must hook the form."""
    add_visit(species="Bushtit")
    add_visit(visitor="raccoon", species=None)
    plain, full = client.get("/?kind=mammal"), client.get("/?species=&kind=mammal&tod=&date_from=&date_to=")
    assert 'aria-current="page">Mammals<' in plain.text and 'aria-current="page">Mammals<' in full.text
    assert "1 matching</h2>" in plain.text and "1 matching</h2>" in full.text
    html = plain.text
    assert 'id="more-filters"' in html and 'a class="clear"' in html
    assert 'f.id !== "more-filters"' in html and '#more-filters a.clear' in html
    assert 'behavior: "instant"' in html  # the page scrolls smoothly; restoring the view must not animate


def test_load_more_pages_in_24s(client, add_visit):
    for _ in range(30):
        add_visit()
    first = client.get("/").text
    assert first.count('class="card"') + first.count('class="card hero-dup"') == 24 and "/visits/more?before=" in first
    more = client.get(more_link(first)).text
    assert more.count('class="card"') == 6 and "load-more" not in more


def test_home_splits_cards_into_today_and_earlier(client, add_visit):
    old = (FROZEN_NOW - timedelta(days=2)).replace(microsecond=0).isoformat()
    for _ in range(20):
        add_visit(captured_at=old)
    for _ in range(26):
        add_visit()
    first = client.get("/").text
    assert first.count("grid-break grid-today") == 1 and "grid-break grid-earlier" not in first  # 24 cards, all from today
    more = client.get(more_link(first)).text
    assert "grid-break grid-today" not in more and more.count("grid-break grid-earlier") == 1
    assert more.index("grid-break grid-earlier") > more.index('class="card"')  # 2 more today first, then Earlier
    assert "load-more" not in more
    assert "grid-break grid-today" not in client.get("/?kind=bird").text  # filters show one plain grid


def test_home_says_no_visitors_yet_today(client, add_visit):
    add_visit(captured_at=(FROZEN_NOW - timedelta(days=1)).replace(microsecond=0).isoformat())
    page = client.get("/").text
    assert "No visitors yet today." in page and page.index("grid-break grid-today") < page.index("grid-break grid-earlier")


def test_home_has_no_motion_funnel(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/").text
    assert "tally-box" not in html and html.count("Visitor today") == 1
    assert "Sent to Claude" not in html and "Kept on the Pi" not in html


def test_images_are_served(client, stored_visit, conn):
    row = db.get_visit(conn, stored_visit())
    r = client.get(f"/images/{row['thumb_path']}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_latest_visit_api_ignores_false_triggers(client, add_visit):
    assert client.get("/api/latest-visit").json() == {"id": 0, "label": None, "captured_at": None, "any_id": 0,
                                                   "waiting": {"review": 0, "false": 0}}
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


def test_favicon_and_static_assets_are_versioned(client):
    assert client.get("/favicon.ico").headers["content-type"] == "image/x-icon"
    html = client.get("/").text
    assert f"style.css?v={webapp.templates.env.globals['asset_version']}" in html


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


def test_latest_visitor_thumbnail_shows_claudes_circle(client, add_visit):
    add_visit(raw_json=json.dumps({"calls": {"circle": {"x": 0.45, "y": 0.65, "r": 0.05, "claude": True}}}))
    html = client.get("/").text
    assert "hero-photo" in html and html.count('class="marks thumb-marks"') == 2 and "0.45" in html  # big photo + grid card


def test_latest_visitor_takes_the_big_photo_and_the_live_view_has_its_own_tab(client, add_visit):
    html = client.get("/").text
    assert 'class="hero-latest"' not in html and 'class="hero hero-empty"' in html  # no visitor yet: Today takes the row
    assert 'id="live-box"' not in html and 'href="/live"' in html  # the live view is on /live, linked from the empty page
    vid = add_visit(species="Bushtit", count=3)
    html = client.get("/").text
    hero = html[html.index('class="hero-latest"'):html.index('class="today"')]
    assert f'href="/visit/{vid}"' in hero and "Bushtit x3" in hero and "_t.jpg" not in hero  # full-size frame
    assert "hero-kicker-top" not in html and hero.index("hero-kicker") < hero.index("hero-name") < hero.index("hero-time")  # one chip, kicker first
    assert 'id="live-box"' not in html and 'id="live-viewer"' not in html and "live-view" not in html.replace("/live", "")


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


def test_only_the_latest_visitor_and_live_view_carry_a_zoom_button(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "PAGE_SIZE", 1)
    first = add_visit(image_path="2026/11/01/a.jpg", thumb_path="2026/11/01/a_t.jpg", species="Bushtit")
    second = add_visit(image_path="2026/11/01/b.jpg", thumb_path="2026/11/01/b_t.jpg", species="Dark-eyed Junco")
    html = client.get("/").text
    hero = html[html.index('class="hero-latest"'):html.index('class="today"')]
    assert 'href="/visit/%d"' % second in hero and 'data-zoom-photo="/images/2026/11/01/b.jpg"' in hero
    assert hero.index("</a>") < hero.index("zoom-btn")  # a sibling of the link, not inside it
    assert html.count('data-zoom-photo="') == 1 and "card-zoom" not in html  # visit cards have none
    assert "live-zoom-btn" not in html and 'id="home-viewer"' in html  # the live view's magnifier is on /live now
    assert "hero-goto" not in hero and "Go to visit" not in html  # the photo link is the only way in
    assert 'id="home-viewer-reset"' in html and "Reset view" in html  # zoom viewer
    # Load more cards and the other pages have none
    more = client.get(more_link(html)).text
    assert f'href="/visit/{first}"' in more and "zoom-btn" not in more
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
    # The installed Android app paints its navigation bar with the manifest's theme_color and has no dark
    # variant, so any color here would be wrong on a phone in the other mode (a white bar in dark mode).
    assert "theme_color" not in m
    assert 'rel="manifest"' in client.get("/").text
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    m = client.get("/manifest.webmanifest").json()
    assert m["start_url"] == "/birdbrain/" and m["scope"] == "/birdbrain/"
    assert all(i["src"].startswith("/birdbrain/static/") for i in m["icons"])
    assert 'href="/birdbrain/manifest.webmanifest' in client.get("/").text


def test_home_lists_real_visits_only(client, add_visit):
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    html = client.get("/").text
    assert html.count('class="card hero-dup"') == 1 and "Bushtit" in html


def nav_of(html: str) -> str:
    return html[html.index("<nav>"):html.index("</nav>")]


def test_nav_visits_tab_is_the_home_page(client, add_visit):
    # The old Visits archive is gone (2026-11-03); the home page is the Visits tab.
    nav = nav_of(client.get("/").text)
    assert '<a href="/" aria-current="page">Visits</a>' in nav
    assert 'href="/visits"' not in nav and ">Latest<" not in nav
    vid = add_visit()
    nav = nav_of(client.get(f"/visit/{vid}").text)
    assert '<a href="/" aria-current="page">Visits</a>' in nav and 'href="/visits"' not in nav
    nav = nav_of(client.get("/species").text)
    assert '<a href="/" >Visits</a>' in nav and nav.count('aria-current="page"') == 1


def test_old_visits_archive_links_redirect_home_with_their_filters(client, add_visit):
    for _ in range(3):
        add_visit(species="Bushtit")
    mammal = add_visit(visitor="raccoon", species=None)
    r = client.get("/visits", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/"
    r = client.get("/visits?kind=mammal&tod=day", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/?kind=mammal&tod=day"
    html = client.get("/visits?kind=mammal").text  # followed: the filtered home page
    assert html.count('class="card"') == 1 and f'href="/visit/{mammal}"' in html and "1 matching</h2>" in html
    # the archive's Load more keeps paging through the home list
    r = client.get("/visits/archive-more?kind=bird&before=2026-11-01T10:00:00,5", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/visits/more?kind=bird&before=2026-11-01T10:00:00,5"
    assert client.get("/visits/archive-more?before=junk").text.count('class="card"') == 4


def test_tour_script_and_rerun_buttons(client):
    assert "/static/tour.js" in client.get("/").text
    about = client.get("/about").text
    assert "data-tour-restart" in about
    assert 'class="popup-close about-tour-close"' in about and "bb-about-tour-off" in about  # the box can be hidden per browser
    assert '<p class="about-tour-mini" hidden><a href="/?tour=1" data-tour-restart' in about  # and a small link takes its place
    assert "data-tour-restart" in client.get("/settings").text
    assert "data-admin=\"1\"" in client.get("/").text  # no password: everyone is admin
    assert "tour.js" not in client.get("/?embed=1").text
    tour = client.get("/static/tour.js").text  # the tour is the same for everyone: no admin steps or flags
    assert 'part: "admin"' not in tour and "bb_tour_admin_seen" not in tour


def test_every_big_photo_view_closes_with_the_minus_magnifier(client, add_visit):
    vid = add_visit(image_path="2026/11/01/a.jpg", thumb_path="2026/11/01/a_t.jpg", species="Bushtit")
    for html in (client.get("/").text, client.get(f"/visit/{vid}").text):
        assert "viewer-zoom-out photo-viewer-close" in html and "&times;" not in html.split('class="viewer')[1]
    assert "viewer-zoom-out live-viewer-close" in client.get("/live").text


def _nav(html: str) -> str:
    return re.search(r"<nav>.*?</nav>", html, re.S).group(0)


def test_nav_shows_waiting_counts_for_admin(client, add_visit):
    for _ in range(3):
        add_visit(needs_review=1)
    for _ in range(2):
        add_visit(visitor="none", species=None)
    nav = _nav(client.get("/").text)
    assert re.search(r'href="/review"[^>]*class="waiting"[^>]*title="3 waiting for review".*?<span class="nav-n"[^>]*>3</span>', nav, re.S)
    assert re.search(r'href="/false-triggers"[^>]*class="waiting"[^>]*title="2 false triggers to check".*?<span class="nav-n"[^>]*>2</span>', nav, re.S)


def test_nav_has_no_bubbles_when_nothing_waits(client, add_visit):
    add_visit()
    nav = _nav(client.get("/").text)
    assert "nav-n" not in nav and "waiting" not in nav


def test_nav_caps_the_count_at_99_plus(client, add_visit):
    for _ in range(100):
        add_visit(needs_review=1)
    assert '>99+</span>' in _nav(client.get("/").text)


def test_nav_counts_not_shown_to_visitors(client, add_visit, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    add_visit(needs_review=1)
    add_visit(visitor="none", species=None)
    assert "nav-n" not in client.get("/").text


def test_latest_visit_api_has_waiting_counts_and_nav_hooks(client, add_visit):
    add_visit(needs_review=1)
    add_visit(visitor="none", species=None)
    assert client.get("/api/latest-visit").json()["waiting"] == {"review": 1, "false": 1}
    nav = _nav(client.get("/").text)
    assert 'data-nav="review"' in nav and 'data-nav="false"' in nav


def test_visits_paging_ties_and_bad_cursor(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "PAGE_SIZE", 2)
    ids = [add_visit(captured_at="2026-11-01T10:00:00") for _ in range(5)]  # the same second: the id breaks ties
    first = client.get("/").text
    second = client.get(more_link(first)).text
    third = client.get(more_link(second)).text
    assert card_ids(first) + card_ids(second) + card_ids(third) == ids[::-1] and "load-more" not in third
    assert card_ids(client.get("/visits/more?before=junk").text) == ids[::-1][:2]


def test_home_has_rotate_hint_button_and_popup(client):
    html = client.get("/").text
    assert 'id="rotate-btn"' in html and 'type="button"' in html
    assert 'aria-label="Turn sideways for a bigger view"' in html
    assert 'id="rotate-dialog"' in html and "Try landscape" in html and "Got it" in html
    assert "rotate-hint.js" in html
    assert "—" not in html.split('id="rotate-dialog"')[1].split("</dialog>")[0]


def test_theme_color_is_set_before_first_paint_and_matches_css(client):
    import re
    from pathlib import Path
    html = client.get("/").text
    # the metas exist before the early script, which sets the one it keeps from the saved theme or the device scheme
    assert html.index('id="theme-color"') < html.index("window.bbBg")
    assert "window.bbTheme.apply();" in html and "prefers-color-scheme: dark" in html
    css = (Path(__file__).resolve().parents[3] / "birdbrain/web/static/style.css").read_text()
    assert re.search(r"^html \{ background: var\(--bg\); \}", css, re.M)
    pairs = re.findall(r'(\w+): \{ light: "(#\w+)", dark: "(#\w+)" \}', html)
    assert [name for name, _, _ in pairs] == ["forest", "ocean", "sunset", "plum"]
    # each palette's light and dark --bg in style.css, by selector
    light_bg = {"forest": re.search(r"^:root \{\s*--bg: (#\w+)", css, re.M).group(1),
                "forest_dark": re.search(r'^:root\[data-theme="dark"\] \{\s*--bg: (#\w+)', css, re.M).group(1)}
    for name, light, dark in pairs:
        if name == "forest":
            assert (light, dark) == (light_bg["forest"], light_bg["forest_dark"])
            continue
        assert re.search(rf':root\[data-palette="{name}"\] \{{\s*--bg: {light};', css), name
        assert re.search(rf':root\[data-palette="{name}"\]\[data-theme="dark"\] \{{\s*--bg: {dark};', css), name
        assert re.search(rf':root\[data-palette="{name}"\]:not\(\[data-theme="light"\]\) \{{\s*--bg: {dark};', css), name


def test_home_chips_are_swappable_in_place(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/?kind=bird").text
    assert 'aria-current="page"' in html and 'id="more-filters"' in html and "bb-home-url" in html


def test_live_tab_sits_right_after_species_and_renders(client, add_visit):
    nav = _nav(client.get("/").text)
    assert nav.index('>Visits<') < nav.index('href="/species"') < nav.index('href="/live"') < nav.index('href="/stats"')
    assert 'href="/live" aria-current="page">Live<' in _nav(client.get("/live").text)
    html = client.get("/live").text
    assert html.count('id="live-box"') == 1 and 'id="live-viewer"' in html and "/static/live-view.js" in html
    assert "0</span>" in html and "Last visitor" not in html  # no visitor yet
    vid = add_visit(species="Bushtit", count=3)
    html = client.get("/live").text
    panel = html[html.index('id="live-visits"'):]
    assert "Last visitor" in panel and f'href="/visit/{vid}"' in panel and "Bushtit" in panel and " x3" in panel
    assert "1</span>" in panel and "1 species" in panel


def test_live_tab_reads_the_light_from_the_frame_not_the_camera(client, monkeypatch):
    from io import BytesIO
    from PIL import Image
    def jpeg(color):
        out = BytesIO(); Image.new("RGB", (64, 36), color).save(out, "JPEG"); return out.getvalue()
    def no_camera(*a, **k):
        raise AssertionError("the page must not call the camera API")
    monkeypatch.setattr(webapp.camera, "_call", no_camera)
    for color, light, other in (((120, 160, 90), "day", "night"), ((100, 100, 100), "night", "day")):
        monkeypatch.setattr(webapp, "_live", {"jpeg": jpeg(color), "at": time.time(), "tried": 0.0})
        html = client.get("/live").text
        assert f'data-light="{light}">' in html and f'data-light="{other}" hidden' in html
        assert client.get("/live/latest.jpg").headers["X-Frame-Light"] == light
    monkeypatch.setattr(webapp, "_live", {"jpeg": b"not a picture", "at": time.time(), "tried": 0.0})
    assert "X-Frame-Light" not in client.get("/live/latest.jpg").headers and "unknown" in client.get("/live").text


def test_keyboard_shortcuts_popup_lists_each_pages_own_keys(client, add_visit):
    vid = add_visit(visitor="bird", species="Song Sparrow")
    rows, heads = _keys_dialog(client.get("/").text)
    assert ("→", "Open the latest visit") in rows and heads == ["This page", "Everywhere"]
    rows, heads = _keys_dialog(client.get(f"/visit/{vid}").text)
    assert [r[1] for r in rows[:5]] == ["Newer visit", "Older visit", "Home (at the top of the page)",
                                        "Back home (if you came by →)",
                                        "Confirm no animal, then next"]  # admin: the C row too
    assert heads == ["This page", "Everywhere"]
    rows, heads = _keys_dialog(client.get("/species/Song%20Sparrow").text)
    assert [r[0] for r in rows[:3]] == ["←", "→", "↑"] and "Species list" in rows[2][1]


def test_home_marks_only_the_heros_card_as_a_repeat(client, add_visit):
    assert 'class="card hero-dup"' not in client.get("/").text  # no visits: nothing marked
    first = add_visit(species="Bushtit")
    html = client.get("/").text
    assert html.count('class="card hero-dup"') == 1 and f'hero-dup" href="/visit/{first}"' in html
    second = add_visit(species="Bushtit")
    html = client.get("/").text
    assert html.count('class="card hero-dup"') == 1 and f'hero-dup" href="/visit/{second}"' in html
    assert 'class="card hero-dup"' not in client.get("/?kind=bird").text  # filtered views have no hero repeat


def test_species_and_home_show_how_long_ago(client, add_visit):
    add_visit(species="Bushtit", captured_at=(FROZEN_NOW - timedelta(minutes=5)).isoformat())
    home = client.get("/").text
    assert re.search(r'class="ago" data-age="300">\(5 min ago\)<', home)
    assert 'src="/static/ago.js' in home
    page = client.get("/species/Bushtit").text
    assert re.search(r'class="ago" data-age="300">\(5 min ago\)<', page)  # the fallback ticks too
    assert 'src="/static/ago.js' in page
