import io
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import FROZEN_NOW
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
        image_rel, thumb_rel = images.store_frame(buf.getvalue(), FROZEN_NOW)
        return add_visit(image_path=image_rel, thumb_path=thumb_rel, **fields)
    return add


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
    assert {"not": {"selector_matches": ".hero-link, .hero-goto"}} in rules["prerender"][0]["where"]["and"]
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


def test_home_lists_visits_with_chips_and_filters_toggle(client, add_visit):
    add_visit(species="Bushtit")
    add_visit(visitor="none", species=None)
    html = client.get("/").text
    assert "Bushtit" in html and "tally-box" in html and "Mammals" in html
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


def more_link(html: str) -> str:
    """The Load more button's URL in a page or fragment."""
    import html as htmllib
    m = re.search(r'class="load-more" hx-get="([^"]+)"', html)
    assert m, "no Load more button"
    return htmllib.unescape(m.group(1))


def test_load_more_pages_in_24s(client, add_visit):
    for _ in range(30):
        add_visit()
    first = client.get("/").text
    assert first.count('class="card"') == 24 and "/visits/more?before=" in first
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
    assert 'data-tip="Previous frame (wraps)"' in html and 'data-tip="Next frame (wraps)"' in html and "/ 3" in html
    # Under the photo (inside its column, after the legend line), not in the details column; no on-photo hint.
    media = html[html.index('class="visit-media"'):html.index('class="viewer photo-viewer"')]
    assert 'class="vburst"' in media and media.index('id="mark-legend"') < media.index('class="vburst"')
    assert "photo-hint" not in html and "mark-hint" not in html
    client.post(f"/visit/{vid}/delete")
    assert images.burst_paths(image_rel) == []


def test_burst_arrows_absent_without_a_burst(client, stored_visit):
    vid = stored_visit(visitor="none", species=None)
    assert 'class="vburst-step"' not in client.get(f"/visit/{vid}").text


@pytest.fixture
def burst_visit(stored_visit):
    """A false trigger with a 4-frame burst on disk; Claude saw frame 2. Returns (id, image_rel, burst paths)."""
    def add(**fields):
        vid = stored_visit(visitor="none", species=None, raw_json=json.dumps({"burst": {"count": 4, "best": 1, "fps": 4}}),
                           **fields)
        with db.session() as conn:
            image_rel = db.get_visit(conn, vid)["image_path"]
        frames = []
        for shade in (40, 90, 140, 190):  # each frame its own gray, so the thumbnail says which frame it came from
            buf = io.BytesIO()
            Image.new("RGB", (640, 360), (shade, shade, shade)).save(buf, "JPEG")
            frames.append(buf.getvalue())
        return vid, image_rel, images.store_burst(frames, image_rel)
    return add


def test_picking_a_burst_frame_as_the_photo_is_admin_only_and_false_triggers_only(client, burst_visit, stored_visit, monkeypatch):
    vid, image_rel, saved = burst_visit()
    real = stored_visit()
    assert client.post(f"/visit/{real}/photo-frame", data={"frame": "1"}).status_code == 400  # one frame only
    for bad in ("5", "-1", "x", "99"):
        assert client.post(f"/visit/{vid}/photo-frame", data={"frame": bad}).status_code == 400, bad
    assert client.post(f"/visit/{vid}/photo-frame", data={"frame": "1"}).headers["HX-Refresh"] == "true"
    assert client.post("/visit/999999/photo-frame", data={"frame": "1"}).status_code == 404
    # Visitors: the change is refused and the page shows no controls, but it does show the picked photo.
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    assert client.post(f"/visit/{vid}/photo-frame", data={"frame": "0"}).status_code == 403
    html = client.get(f"/visit/{vid}").text
    assert 'id="use-frame"' not in html and 'id="use-claude"' not in html and "photo-frame" not in html
    assert f'src="/images/{saved[0]}"' in html and "your photo" in html
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


def test_circles_keep_the_burst_frame_they_were_drawn_on():
    marks = json.dumps([
        {"x": 0.5, "y": 0.5, "r": 0.05},                          # on the photo Claude saw
        {"x": 0.2, "y": 0.3, "r": 0.05, "frame": 4},              # on burst frame 4
        {"x": 0.2, "y": 0.3, "r": 0.05, "frame": 2},              # frame 2 is the photo Claude saw: no frame
        {"x": 0.1, "y": 0.1, "r": 0.05, "frame": 0},              # not a frame
        {"x": 0.1, "y": 0.1, "r": 0.05, "frame": "3"},            # not a number
        {"x": 0.1, "y": 0.1, "r": 0.05, "frame": True},           # not a number either
        {"x": 0.1, "y": 0.1, "r": 0.05, "frame": 999},            # no burst is that long
        {"x": 0.3, "y": 0.3, "r": 0.05, "frame": 5, "claude": True},  # Claude's circles stay on its photo
    ] + [{"x": 0.9, "y": 0.9, "r": 0.02, "frame": 6}] * 4)
    got = db.parse_marks(marks, claude_frame=2)
    assert len(got) == 9  # still at most 9, frames or not
    assert [m.get("frame") for m in got[:8]] == [None, 4, None, None, None, None, None, None]
    assert got[7] == {"x": 0.3, "y": 0.3, "r": 0.05, "claude": True} and got[8]["frame"] == 6
    assert db.parse_marks(marks)[2]["frame"] == 2  # without knowing Claude's frame it stays
    assert db.claude_frame(json.dumps({"burst": {"best": 1}})) == 2 and db.claude_frame("{}") is None
    assert "circle 2 at 20% across, 30% down (on burst frame 4, not the photo you saw)" in db.describe_marks(got)


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


def test_decision_steps_handles_refusals_and_old_or_broken_records():
    refused = json.dumps({"calls": {"haiku": {"answer": None}, "sonnet": {"frames": 3, "answer": None}}})
    steps = webapp.decision_steps(refused)["steps"]
    assert [s["refused"] for s in steps] == [True, True]
    assert [s["error"] for s in steps] == [None, None] and not webapp.decision_steps(refused)["failed"]


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
    assert html.index("Visitor today") < html.index("More filters")
    stats = client.get("/stats").text
    rows = ["False alarms", "Kept on the Pi", "Sent to Claude"]
    assert "Today's Visits" in stats and "Today's motion" not in stats
    assert 'id="presence"' not in stats and "#presence" not in stats
    assert "visits</span>" in stats and stats.index("Today's Visits") < stats.index("Most frequent")
    start = stats.index("New to the yard")
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


def _seed_filter_species(add_visit):
    """Junco visits: a day one today, a night one today, a two-bird one, an old one; plus a Bushtit that must never show."""
    today = FROZEN_NOW.replace(microsecond=0)
    add_visit(species="Dark-eyed Junco", captured_at=today.isoformat())
    add_visit(species="Dark-eyed Junco", captured_at=(today - timedelta(minutes=5)).isoformat(), is_night=1)
    add_visit(species="Dark-eyed Junco", count=2, captured_at=(today - timedelta(minutes=9)).isoformat())
    add_visit(species="Dark-eyed Junco", captured_at=(today - timedelta(days=20)).isoformat())
    add_visit(species="Bushtit", also=json.dumps([{"visitor": "bird", "species": "Dark-eyed Junco", "count": 1, "confidence": "high"}]))
    add_visit(species="Bushtit")


def _cards(html):
    import re
    return re.findall(r'<a class="card" href="/visit/(\d+)', html)


def test_species_visits_tab_has_the_home_filters_minus_species_and_kinds(client, add_visit):
    _seed_filter_species(add_visit)
    html = client.get("/species/Dark-eyed%20Junco").text
    panel = html[html.index('id="panel-visits"'):]
    assert 'class="filters-toggle"' in panel and 'id="more-filters"' in panel and "/static/species-filters.js" in html
    assert panel.index('class="chips"') < panel.index('id="more-filters"') < panel.index('id="visit-grid"')
    chips = panel[panel.index('class="chips"'):panel.index("</nav>")]
    for label in ("All", "Multiple", "Night", "Today", "This week"):
        assert f">{label}<" in chips
    assert ">Birds<" not in chips and ">Mammals<" not in chips
    form = panel[panel.index('id="more-filters"'):panel.index("</form>")]
    assert 'name="tod"' in form and 'name="date_from"' in form and 'name="date_to"' in form and 'name="species"' not in form
    assert "data-filtered" not in panel[:panel.index(">")] and ">5</span> visits" in panel and "(5)" in html
    assert len(_cards(html)) == 5  # the Bushtit photo with a junco in it counts as a junco visit


def test_species_visits_filter_by_night_multiple_and_dates_and_open_on_the_visits_tab(client, add_visit):
    _seed_filter_species(add_visit)
    night = client.get("/species/Dark-eyed%20Junco?tod=night").text
    assert len(_cards(night)) == 1 and 'aria-current="page">Night<' in night and ">1</span> of 5 visits" in night
    assert "(1 of 5)" in night and 'id="panel-visits"' in night and "data-filtered" in night and ">Clear<" in night
    day = client.get("/species/Dark-eyed%20Junco?tod=day").text
    assert len(_cards(day)) == 4 and 'id="more-filters"' in day and 'aria-expanded="true"' in day
    multi = client.get("/species/Dark-eyed%20Junco?kind=multiple").text
    assert len(_cards(multi)) == 2 and 'aria-current="page">Multiple<' in multi  # the two-bird visit and the Bushtit with a junco too
    week = client.get("/species/Dark-eyed%20Junco?date_from=2026-01-01&date_to=2026-01-02").text
    assert "No visits match these filters." in week and "Show all" in week and 'id="visit-grid"' not in week
    # Birds / Mammals and a species in the URL don't apply on a species page.
    assert len(_cards(client.get("/species/Dark-eyed%20Junco?kind=mammal&species=Bushtit").text)) == 5


def test_species_filter_chips_keep_the_sort_and_end_in_the_visits_hash(client, add_visit):
    _seed_filter_species(add_visit)
    html = client.get("/species/Dark-eyed%20Junco?sort=name&tod=night").text
    assert 'href="/species/Dark-eyed%20Junco?tod=night&amp;sort=name#visits"' in html
    assert 'href="/species/Dark-eyed%20Junco?sort=name#visits"' in html  # All
    assert 'name="sort" value="name"' in html
    assert _swipe_links(html)["prev"] is None or "sort=name" in _swipe_links(html)["prev"]


def test_species_load_more_keeps_the_filters(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "SPECIES_PAGE", 2)
    _seed_filter_species(add_visit)
    first = client.get("/species/Dark-eyed%20Junco?tod=day").text
    assert len(_cards(first)) == 2
    import html as htmllib, re
    url = htmllib.unescape(re.search(r'hx-get="(/species/[^"]*/more[^"]*)"', first).group(1))
    assert "tod=day" in url and "before=" in url
    page2 = client.get(url).text
    got = _cards(first) + _cards(page2)
    assert len(got) == 4 and len(set(got)) == 4  # all four day visits, none of the night one, no repeats
    assert len(_cards(client.get("/species/Dark-eyed%20Junco/more?tod=night").text)) == 1


def _seed_swipe_species(add_visit):
    # visits order: Bushtit (3), Song Sparrow (2), Anna's Hummingbird (1); A to Z: Anna's, Bushtit, Song
    for _ in range(3):
        add_visit(species="Bushtit", captured_at="2026-10-01T09:00:00")
    for _ in range(2):
        add_visit(species="Song Sparrow", captured_at="2026-10-05T09:00:00")
    add_visit(species="Anna's Hummingbird", captured_at="2026-10-09T09:00:00")


def _swipe_links(html):
    import re
    return {rel: (m.group(1) if (m := re.search(rf'<link rel="{rel}" href="([^"]*)"', html)) else None) for rel in ("prev", "next")}


def test_species_page_swipes_along_the_list_order(client, add_visit):
    _seed_swipe_species(add_visit)
    html = client.get("/species/Song%20Sparrow").text
    assert 'data-swipe="links"' in html
    assert _swipe_links(html) == {"prev": "/species/Bushtit", "next": "/species/Anna%27s%20Hummingbird"}
    # the ends have only one neighbour
    assert _swipe_links(client.get("/species/Bushtit").text) == {"prev": None, "next": "/species/Song%20Sparrow"}
    assert _swipe_links(client.get("/species/Anna's%20Hummingbird").text)["next"] is None


def test_species_page_swipe_follows_the_lists_sort(client, add_visit):
    _seed_swipe_species(add_visit)
    html = client.get("/species/Bushtit?sort=name").text
    assert _swipe_links(html) == {"prev": "/species/Anna%27s%20Hummingbird?sort=name", "next": "/species/Song%20Sparrow?sort=name"}
    # newest first seen: Anna's, Song, Bushtit
    assert _swipe_links(client.get("/species/Song%20Sparrow?sort=newest").text)["prev"] == "/species/Anna%27s%20Hummingbird?sort=newest"
    # an unknown sort is the default order, with clean links
    assert _swipe_links(client.get("/species/Song%20Sparrow?sort=bogus").text)["prev"] == "/species/Bushtit"


def test_species_order_matches_the_list(conn, add_visit):
    _seed_swipe_species(add_visit)
    add_visit(species="Bushtit", visitor="bird", also=json.dumps([{"visitor": "bird", "species": "Steller's Jay", "count": 1, "confidence": "high"}]))
    for sort in db.SPECIES_SORTS:
        assert db.species_order(conn, sort) == [r["label"] for r in db.species_summary(conn, sort=sort)]


def test_species_list_tiles_carry_the_sort(client, add_visit):
    _seed_swipe_species(add_visit)
    assert 'href="/species/Bushtit"' in client.get("/species").text
    assert 'href="/species/Bushtit"' in client.get("/species?sort=visits").text
    assert 'href="/species/Bushtit?sort=name"' in client.get("/species?sort=name").text
    assert "?sort=bogus" not in client.get("/species?sort=bogus").text.replace('href="?sort=', "")


def test_species_page_has_the_full_size_photo_browser(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/species/Bushtit").text
    assert 'class="lb-open"' in html and "/static/lightbox.js" in html
    assert 'data-full="/images/' in html


@pytest.mark.parametrize("path", ["/stats", "/review", "/false-triggers", "/species?sort=name", "/species?sort=bogus"])
def test_pages_render(client, add_visit, path):
    add_visit(needs_review=1)
    assert client.get(path).status_code == 200


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
    steps = ["(Anything there?)", "(What species?)", "3</span> Claude"]
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


def test_visit_page_says_which_zone_saw_the_motion(client, add_visit):
    vid = add_visit(species="Virginia Opossum", visitor="possum", raw_json=json.dumps({"trigger": "ground", "calls": {
        "haiku": {"answer": {"visitor": "possum", "species": "Virginia Opossum", "confidence": "high"}}}}))
    assert "Spotted moving on the ground." in client.get(f"/visit/{vid}").text
    assert webapp.decision_steps(json.dumps({"calls": {}}))["trigger"] is None   # --once captures have none


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


def test_false_triggers_come_in_pages(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "FT_PAGE", 2)
    for _ in range(3):
        add_visit(visitor="none", species=None)
    html = client.get("/false-triggers").text
    assert html.count('class="card"') == 2 and "/false-triggers/more?before=" in html
    more = client.get(more_link(html)).text
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


def test_species_list_names_open_the_info_card_and_card_has_no_jump_to_visits(client, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    assert 'data-species="Bushtit"' in client.get("/species").text
    from birdbrain import species_info
    monkeypatch.setattr(species_info, "get", lambda conn, name, fetch=True: {"name": name, "common": name, "status": "Year-round resident"})
    html = client.get("/species-info/Bushtit").text
    assert "Jump to visits" not in html and "sp-jump" not in html
    assert "All visits" not in html and 'class="sp-facts"' in html
    assert 'id="visit-grid"' in client.get("/species/Bushtit").text
    page = client.get("/species/Bushtit").text
    for tab in re.findall(r'data-sp-tab="(\w+)"', page):  # a tab name in the URL hash must not match an element id, or the browser scrolls to it
        assert f'id="{tab}"' not in page
    assert re.findall(r'class="badge [^"]*"(?! data-tip)', html) == []


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
    assert f'<a class="hero-goto" href="/visit/{second}" title="Open this visit (→)">Go to visit</a>' in hero  # after the kicker, a real link
    assert hero.index("hero-kicker") < hero.index("hero-goto") < hero.index("hero-name")
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
    assert html.count('class="card"') == 1 and "Bushtit" in html


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
    about = client.get("/about").text
    assert "data-tour-restart" in about
    assert 'class="popup-close about-tour-close"' in about and "bb-about-tour-off" in about  # the box can be hidden per browser
    assert '<p class="about-tour-mini" hidden><a href="/?tour=1" data-tour-restart' in about  # and a small link takes its place
    assert "data-tour-restart" in client.get("/settings").text
    assert "data-admin=\"1\"" in client.get("/").text  # no password: everyone is admin
    assert "tour.js" not in client.get("/?embed=1").text


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


def test_false_trigger_cards_after_returns_only_newer_false_triggers(client, add_visit, conn):
    old = add_visit(captured_at="2026-10-24T09:00:00", visitor="none", species=None)
    real = add_visit(captured_at="2026-10-24T10:00:00")
    new = add_visit(captured_at="2026-10-24T11:00:00", visitor="none", species=None)
    done = add_visit(captured_at="2026-10-24T12:00:00", visitor="none", species=None)
    db.confirm_visits(conn, [done])
    r = client.get(f"/false-triggers/cards?after={old}")
    assert f'data-id="{new}"' in r.text and f'data-id="{old}"' not in r.text
    assert f'data-id="{real}"' not in r.text and f'data-id="{done}"' not in r.text
    assert r.headers["X-Unchecked"] == "2" and r.headers["X-Total"] == "3"
    shown = client.get(f"/false-triggers/cards?after={old}&show=all").text
    assert f'data-id="{done}"' in shown and f'data-id="{new}"' in shown
    assert 'data-id="' not in client.get(f"/false-triggers/cards?after={done}").text


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


def card_ids(html: str) -> list[int]:
    return [int(i) for i in re.findall(r'class="card" href="/visit/(\d+)', html)]


def test_false_trigger_paging_survives_confirming_between_pages(client, add_visit, conn, monkeypatch):
    """Confirmed photos leave the list; the next page must still start right after the last card seen."""
    monkeypatch.setattr(webapp, "FT_PAGE", 3)
    ids = [add_visit(visitor="none", species=None, captured_at=f"2026-11-01T10:00:{i:02d}") for i in range(9)]
    newest_first = ids[::-1]
    first = client.get("/false-triggers").text
    assert card_ids(first) == newest_first[:3]
    db.confirm_visits(conn, newest_first[:3])  # the owner confirms the page just seen
    second = client.get(more_link(first)).text
    assert card_ids(second) == newest_first[3:6]
    db.confirm_visits(conn, newest_first[3:5])  # and two of the next page, which leave the list as well
    third = client.get(more_link(second)).text
    assert card_ids(third) == newest_first[6:9]
    assert card_ids(client.get(more_link(third)).text) == []  # a full page can be the last: the next one is just empty
    shown = client.get("/false-triggers?show=all").text
    assert "show=all" in more_link(shown) and card_ids(client.get(more_link(shown)).text) == newest_first[3:6]


def test_visits_paging_ties_and_bad_cursor(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "PAGE_SIZE", 2)
    ids = [add_visit(captured_at="2026-11-01T10:00:00") for _ in range(5)]  # the same second: the id breaks ties
    first = client.get("/").text
    second = client.get(more_link(first)).text
    third = client.get(more_link(second)).text
    assert card_ids(first) + card_ids(second) + card_ids(third) == ids[::-1] and "load-more" not in third
    assert card_ids(client.get("/visits/more?before=junk").text) == ids[::-1][:2]


def test_species_page_without_a_card_has_title_facts_and_charts(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/species/Bushtit").text
    assert "<h1>Bushtit</h1>" in html and 'class="facts' in html and "First seen" not in html and "grid-count" in html
    assert 'id="daily"' in html and 'id="hourly"' in html


def test_species_page_while_the_card_loads_keeps_the_fallback(client, add_visit):
    add_visit(species="Bushtit")  # a species with a card (species_info.has_card), not cached yet
    html = client.get("/species/Bushtit").text
    # The card arrives over htmx and replaces these; if it comes back empty they stay.
    assert 'hx-get="/species-info/Bushtit?fold=1"' in html
    assert 'id="sp-fallback"' in html and "<h1>Bushtit</h1>" in html and 'id="daily"' in html


def test_species_page_pages_past_200_visits(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "SPECIES_PAGE", 4)
    ids = [add_visit(species="Bushtit", captured_at=f"2026-11-01T10:00:{i:02d}") for i in range(10)]
    add_visit(species="Song Sparrow")
    newest_first = ids[::-1]
    first = client.get("/species/Bushtit").text
    assert card_ids(first) == newest_first[:4] and "/species/Bushtit/more?before=" in first
    second = client.get(more_link(first)).text
    assert card_ids(second) == newest_first[4:8] and "?species=Bushtit" in second
    third = client.get(more_link(second)).text
    assert card_ids(third) == newest_first[8:] and "load-more" not in third


def test_review_heading_shows_the_real_total(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "REVIEW_SHOWN", 2)
    for i in range(3):
        add_visit(needs_review=1, captured_at=f"2026-11-01T10:00:{i:02d}")
    html = client.get("/review").text
    assert 'Review queue <span class="muted">3</span>' in html and html.count('class="card"') == 2
    assert "Showing the oldest 2 of 3" in html


FOUR = ("Motion detector", "Pi animal check", "BioCLIP 2 species check", "Claude")


def test_the_false_triggers_intro_shows_the_four_components_with_live_numbers(client):
    from birdbrain import classifier, db
    html = client.get("/false-triggers").text
    strips = re.findall(r'<div class="hs-strip" id="step-\d">', html)
    assert len(strips) == 4 and all(n in html for n in FOUR)
    assert f"{config.TRIGGER_FRAMES} frames in a row" in html and f"{classifier.GATE_MIN_SCORE * 100:g}%" in html
    assert f"newest {db.LESSON_LIMIT} lessons" in html and "<summary>More</summary>" in html
    assert "delete confirmed photos to save space" in html and 'id="ft-delete-link"' in html
    # The strips live in a pop-up opened from a link; the page keeps a one-line intro.
    dialog = html[html.index('<dialog class="popup ft-trains-popup"'):html.index("</dialog>")]
    assert 'id="ft-trains-link"' in html and ">What trains what?<" in html
    assert all(n in dialog for n in FOUR) and len(re.findall(r'class="hs-strip"', dialog)) == 4
    assert "Bottom line" in dialog and "popup-close" in dialog and "Escape" not in html.split("<dialog")[0]
    assert "Motion, but no animal found. Open each photo to check it." in html
    assert 'class="hs-strip"' not in html.replace(dialog, "")


def test_about_has_the_four_steps_as_numbered_cards_with_live_numbers(client):
    from birdbrain import classifier, db, lessons
    html = client.get("/about").text
    start = html.index('id="handled"')
    section = html[start:html.index("</section>", start)]
    assert "How a photo is handled" in section and len(re.findall(r'class="hs-step"', section)) == 4
    for name in FOUR:
        assert f"What the {name} does" in section and f"How the {name} is trained" in section
    assert "What you control" in section
    assert f"{config.TRIGGER_FRAMES} frames in a row" in section and f"{classifier.GATE_MIN_SCORE * 100:g}%" in section
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


def test_false_triggers_page_with_nothing_unchecked_has_no_count_or_confirm_all(client, conn, add_visit):
    vid = add_visit(visitor="none", species=None)
    html = client.get("/false-triggers").text
    assert 'id="ft-unchecked"' in html and "ft-confirm" in html
    db.confirm_visits(conn, [vid])
    html = client.get("/false-triggers").text
    assert 'id="ft-unchecked"' not in html
    assert 'id="ft-confirm"' not in html and "Confirm all shown" not in html
    assert "Nothing left to check" in html


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


def test_embedded_visit_page_does_not_swipe(client, add_visit):
    vid = add_visit()
    html = client.get(f"/visit/{vid}?embed=1").text
    assert "data-swipe=" not in html and "swipe.js" not in html


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
    css = (Path(__file__).resolve().parent.parent / "birdbrain/web/static/style.css").read_text()
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
    root = Path(__file__).resolve().parent.parent / "birdbrain/web"
    js = (root / "static/theme.js").read_text()
    for hook in ('addEventListener("pageshow", refresh)', '"prerenderingchange", refresh', '"visibilitychange"',
                 '"fullscreenchange"', 'addEventListener("storage"', '"htmx:afterSettle"'):
        assert hook in js, hook
    assert "window.bbTheme.nudge()" in js and "window.bbTheme.apply(read)" in js
    base = (root / "templates/base.html").read_text()
    # one meta kept, the rest removed; the nudge restores the real color and drops its inline background
    assert 'querySelectorAll(\'meta[name="theme-color"]\').forEach(x => { if (x !== m) x.remove(); })' in base
    assert 'root.style.removeProperty("background-color")' in base and "document.prerendering" in base


def test_visit_page_back_link_replaces_all_visits(client, add_visit):
    vid = add_visit(species="Bushtit")
    html = client.get(f"/visit/{vid}").text
    assert "All visits" not in html
    assert html.count('href="/" data-back-home>') == 2 and "<span>Back</span>" in html


def test_home_chips_are_swappable_in_place(client, add_visit):
    add_visit(species="Bushtit")
    html = client.get("/?kind=bird").text
    assert 'aria-current="page"' in html and 'id="more-filters"' in html and "bb-home-url" in html


def test_confidence_pills_carry_the_claude_mark(client, add_visit):
    """A high/medium/low pill is Claude's answer, so it gets the mark; Needs review reuses conf-low without it."""
    vid = add_visit(confidence="low", needs_review=1)
    html = client.get(f"/visit/{vid}").text
    assert re.search(r'class="badge conf-low by-claude"[^>]*><svg class="claude-mark"', html)
    assert re.search(r'class="badge conf-low"[^>]*>Needs review', html)


def test_species_title_has_a_play_button_only_when_the_card_has_sounds(client, add_visit, monkeypatch):
    add_visit(species="Bushtit")
    from birdbrain import species_info
    card = {"name": "Bushtit", "common": "Bushtit", "status": "Year-round resident", "links": [],
            "sounds": [{"label": "Call", "file": "https://xc.example/1.mp3", "recordist": "A", "license": "CC", "km": None, "id": "1"},
                       {"label": "Song", "file": "https://xc.example/2.mp3", "recordist": "B", "license": "CC", "km": 40, "id": "2"}]}
    monkeypatch.setattr(species_info, "get", lambda conn, name, fetch=True: dict(card))
    html = client.get("/species-info/Bushtit?fold=1").text
    assert 'class="sp-play"' in html and 'data-src="https://xc.example/2.mp3"' in html and "Song</span>" in html
    assert 'aria-pressed="false"' in html and "by B" in html
    assert html.count('class="sp-play"') == 2 and "Call</span>" in html
    assert html.index("Call</span>") < html.index("Song</span>")  # the list's order
    card["sounds"] = card["sounds"] + [{"label": "Song", "file": "https://xc.example/3.mp3", "recordist": "C", "license": "CC", "km": 9, "id": "3"}]
    html = client.get("/species-info/Bushtit?fold=1").text
    assert html.count('class="sp-play"') == 3 and "Song 1</span>" in html and "Song 2</span>" in html and "Call</span>" in html
    card["sounds"] = card["sounds"][:2]
    mini = client.get("/species-info/Bushtit?mini=1").text
    assert "sp-play" not in mini
    card["sounds"] = [card["sounds"][0]]
    assert 'data-src="https://xc.example/1.mp3"' in client.get("/species-info/Bushtit?fold=1").text
    card["sounds"] = []
    assert "sp-play" not in client.get("/species-info/Bushtit?fold=1").text


def test_live_tab_sits_right_after_visits_and_renders(client, add_visit):
    nav = _nav(client.get("/").text)
    assert nav.index('>Visits<') < nav.index('href="/live"') < nav.index('href="/species"')
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
