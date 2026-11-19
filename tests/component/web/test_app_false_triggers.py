"""Component tests of the web app through TestClient on a temp database: false triggers."""

import json
import re

from helpers.web import FOUR, card_ids, more_link
from birdbrain import config, db
from birdbrain.web import app as webapp


def test_every_status_pill_explains_itself(client, add_visit):
    add_visit(visitor="none", species=None, confidence="medium", is_night=1)
    vid = add_visit(confidence="low", needs_review=1)
    for path in ("/false-triggers", "/", f"/visit/{vid}", "/about"):
        html = client.get(path).text
        pills = html.count('class="badge ')
        assert pills and len(re.findall(r'class="badge [^"]*" data-tip=', html)) == pills, path
    assert "Night: taken in infrared" in client.get("/false-triggers").text


def test_visit_page_arrows_step_through_false_triggers(client, add_visit):
    ft_old = add_visit(captured_at="2026-10-24T09:00:00", visitor="none", species=None)
    add_visit(captured_at="2026-10-24T10:00:00")
    ft_new = add_visit(captured_at="2026-10-24T11:00:00", visitor="none", species=None)
    # In the list of what is left to check the cards carry their list; with Show confirmed, every false trigger.
    assert f'href="/visit/{ft_new}?pool=false"' in client.get("/review?only=false").text
    assert f'href="/visit/{ft_new}?ft=1"' in client.get("/review?only=false&show=all").text
    html = client.get(f"/visit/{ft_new}").text
    assert f'href="/visit/{ft_old}?ft=1" id="nav-next"' in html and 'id="nav-prev"' not in html


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
    assert r.status_code == 303 and r.headers["location"] == "/review?only=false&cleared=1"
    html = client.get("/false-triggers?cleared=1").text
    assert "Deleted 1 photo. The Pi keeps what it learned from them." in html
    assert "without photos" not in html and f'href="/visit/{vid}?' not in html
    assert db.count_visits(conn, false_triggers=True) == 3  # the record stays
    page = client.get(f"/visit/{vid}").text
    assert "Photo deleted to save space" in page and "/images/" + db.get_visit(conn, vid)["image_path"] not in page


def test_false_triggers_come_in_pages(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "FT_PAGE", 2)
    for _ in range(3):
        add_visit(visitor="none", species=None)
    html = client.get("/false-triggers").text
    assert html.count('class="card"') == 2 and "/review/more?before=" in html
    more = client.get(more_link(html)).text
    assert more.count('class="card"') == 1 and "?pool=false" in more and "load-more" not in more


def test_false_trigger_cards_refresh_after_the_pop_up(client, add_visit, conn):
    keep = add_visit(visitor="none", species=None)
    real = add_visit(visitor="none", species=None, corrected_visitor="bird", corrected_species="Bushtit")
    db.confirm_visits(conn, [keep])
    # Confirmed: it has left the list of what is still to check, so the page lets its card go.
    r = client.get(f"/review/cards?ids={keep},{real},abc&only=false")
    assert 'data-id="' not in r.text and json.loads(r.headers["X-Counts"])["false"] == 0
    # With Show confirmed it stays, with its pill; one that turned out to be a real visit is gone either way.
    r = client.get(f"/review/cards?ids={keep},{real},abc&only=false&show=all")
    assert f'data-id="{keep}"' in r.text and f'data-id="{real}"' not in r.text
    assert "Confirmed No Animal" in r.text


def test_false_trigger_cards_after_returns_only_newer_false_triggers(client, add_visit, conn):
    old = add_visit(captured_at="2026-10-24T09:00:00", visitor="none", species=None)
    real = add_visit(captured_at="2026-10-24T10:00:00")
    new = add_visit(captured_at="2026-10-24T11:00:00", visitor="none", species=None)
    done = add_visit(captured_at="2026-10-24T12:00:00", visitor="none", species=None)
    db.confirm_visits(conn, [done])
    r = client.get(f"/review/cards?after={old}&only=false")
    assert f'data-id="{new}"' in r.text and f'data-id="{old}"' not in r.text
    assert f'data-id="{real}"' not in r.text and f'data-id="{done}"' not in r.text
    assert json.loads(r.headers["X-Counts"])["false"] == 2 and r.headers["X-Total"] == "3"
    shown = client.get(f"/review/cards?after={old}&only=false&show=all").text
    assert f'data-id="{done}"' in shown and f'data-id="{new}"' in shown
    assert 'data-id="' not in client.get(f"/review/cards?after={done}&only=false").text


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


def test_the_false_triggers_intro_shows_the_four_components_with_live_numbers(client):
    from birdbrain import classifier, db
    html = client.get("/false-triggers").text
    strips = re.findall(r'<div class="hs-strip" id="step-\d">', html)
    assert len(strips) == 4 and all(n in html for n in FOUR)
    assert f"{config.TRIGGER_FRAMES} frames in a row" in html and f"{classifier.GATE_MIN_SCORE_BOWL:.0%}" in html
    assert f"{classifier.GATE_MIN_SCORE_TIGHT:.0%}" in html
    assert f"newest {db.LESSON_LIMIT} lessons" in html and "<summary>More</summary>" in html
    assert "delete confirmed photos to save space" in html and 'id="ft-delete-link"' in html
    # The strips live in a pop-up opened from a link; the page keeps a one-line intro.
    dialog = html[html.index('<dialog class="popup ft-trains-popup"'):html.index("</dialog>")]
    assert 'id="ft-trains-link"' in html and ">What trains what?<" in html
    assert all(n in dialog for n in FOUR) and len(re.findall(r'class="hs-strip"', dialog)) == 4
    assert "Bottom line" in dialog and "popup-close" in dialog and "Escape" not in html.split("<dialog")[0]
    assert "Motion, but no animal found. Open each photo to check it." in html
    assert 'class="hs-strip"' not in html.replace(dialog, "")


def test_false_triggers_page_with_nothing_unchecked_has_no_count_or_confirm_all(client, conn, add_visit):
    vid = add_visit(visitor="none", species=None)
    html = client.get("/false-triggers").text
    assert 'id="ft-unchecked"' in html and "ft-confirm" in html
    db.confirm_visits(conn, [vid])
    html = client.get("/false-triggers").text
    assert 'id="ft-unchecked"' not in html
    assert 'id="ft-confirm"' not in html and "Confirm all shown" not in html
    assert "Nothing left to check" in html


def test_last_confirm_with_autoplay_on_sends_the_popup_page_home(client, add_visit):
    add_visit(visitor="none")
    page = client.get("/false-triggers").text
    assert 'e.data.home) location.href = "/"' in page
    # The visit page reports home only when it was the last photo and Auto-play is remembered on.
    vid = add_visit(visitor="none")
    visit = client.get(f"/visit/{vid}?todo=1&embed=1").text
    assert 'localStorage.getItem("birdbrain.burstAuto") === "1"' in visit and "last: !next, home" in visit


DONE = "2026-11-02T08:00:00"


def _pool(add_visit):
    """One of each, oldest first: a false trigger, a confirmed bird, a visit waiting for review, a corrected
    bird, an unconfirmed bird."""
    empty = add_visit(captured_at="2026-11-01T09:00:00", visitor="none", species=None)
    add_visit(captured_at="2026-11-01T10:00:00", species="Bushtit", confirmed_at=DONE)
    unsure = add_visit(captured_at="2026-11-01T11:00:00", confidence="low", needs_review=1)
    add_visit(captured_at="2026-11-01T12:00:00", corrected_visitor="bird", corrected_species="Bushtit")
    bird = add_visit(captured_at="2026-11-01T13:00:00", species="Bushtit")
    return empty, unsure, bird


def test_review_is_one_pool_with_a_chip_per_list(client, add_visit):
    empty, unsure, bird = _pool(add_visit)
    html = client.get("/review").text
    # Everything still to do, newest first, whatever its kind; nothing confirmed or corrected.
    assert card_ids(html) == [bird, unsure, empty] and f'href="/visit/{bird}?pool=all"' in html
    chips = html[html.index('<nav class="chips review-chips"'):]
    chips = chips[:chips.index("</nav>")]
    assert re.findall(r'data-list="(\w+)"[^>]*>([^<]+)<span class="chip-n">(\d+)</span>', chips) == [
        ("all", "All", "3"), ("review", "Needs review", "1"), ("false", "False triggers", "1"),
        ("visits", "Confirm visits", "1")]
    assert 'href="/review" data-list="all" aria-current="page"' in chips
    for only, vid in (("review", unsure), ("false", empty), ("visits", bird)):
        page = client.get(f"/review?only={only}").text
        assert card_ids(page) == [vid] and f'href="/visit/{vid}?pool={only}"' in page
        assert f'href="/review?only={only}" data-list="{only}" aria-current="page"' in page
    # The False triggers list alone keeps its tools; the pop-up is on every list.
    assert 'id="ft-storage"' not in html and "Confirm all shown" not in html and 'id="ft-visit"' in html
    assert 'id="ft-storage"' in client.get("/review?only=false").text
    assert "needs review" in html and card_ids(client.get("/review?only=bogus").text) == [bird, unsure, empty]


def test_the_old_false_triggers_address_opens_that_list(client):
    r = client.get("/false-triggers?show=all&x=1", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/review?only=false&show=all"


def test_confirm_from_the_pool_only_lands_on_one_still_to_do(client, add_visit, conn):
    empty, unsure, bird = _pool(add_visit)
    # From the newest, Older skips the corrected bird; from there it skips the confirmed one, to the false trigger.
    html = client.get(f"/visit/{bird}?pool=all&embed=1").text
    assert f'href="/visit/{unsure}?pool=all&amp;embed=1" id="nav-next"' in html and 'id="nav-prev"' not in html
    # Right of the arrows, the pop-up says how many of the list are left (this one included).
    assert re.search(r'id="nav-next"[^>]*>[^<]*</a>\s*<span class="visit-nav-left" id="pool-left"[^>]*><b>3</b> left to review</span>', html)
    html = client.get(f"/visit/{unsure}?pool=all&embed=1").text
    assert f'href="/visit/{empty}?pool=all&amp;embed=1" id="nav-next"' in html
    assert f'href="/visit/{bird}?pool=all&amp;embed=1" id="nav-prev"' in html
    # One list on its own stays inside it: nothing else waits for review, so the pop-up closes after this one.
    html = client.get(f"/visit/{unsure}?pool=review&embed=1").text
    assert 'id="nav-next"' not in html and 'id="nav-prev"' not in html and "else if (true) close();" in html
    assert "<b>1</b> left to review" in html
    # Phones hide the arrows row, so the count also sits in the title line, just before Close.
    assert re.search(r'class="m-only title-left" id="pool-left-m"[^>]*><b>1</b> left</span><a class="m-only title-back embed-close"', html)
    # Confirmed, it has left the pool, and its page still knows what comes next.
    assert client.post(f"/visit/{unsure}/reviewed").status_code == 200
    assert db.pool_counts(conn) == {"review": 0, "false": 1, "visits": 1, "all": 2}
    page = client.get(f"/visit/{unsure}?pool=all").text
    assert f'href="/visit/{empty}?pool=all" id="nav-next"' in page
    assert 'id="pool-left"' not in page and 'id="pool-left-m"' not in page and "Review: All (2)" in page  # in a tab of its own the bar's link counts
    assert "<b>2</b> left to review" in client.get(f"/visit/{empty}?pool=all&embed=1").text
    # In a tab of its own: the bar names the list, and Confirm or C goes on to the next, then back to the list.
    page = client.get(f"/visit/{empty}?pool=false").text
    assert 'href="/review?only=false"' in page and "Review: False triggers (1)" in page
    assert 'location.href = next ? next.href : "/review?only=false"' in page


def test_the_pool_is_the_owners_alone(client, add_visit, locked):
    old = add_visit(captured_at="2026-11-01T09:00:00", species="Bushtit", confirmed_at=DONE)
    new = add_visit(captured_at="2026-11-01T10:00:00", species="Bushtit")
    # A visitor's ?pool is dropped: the arrows walk the visits as always, and nothing mentions the Review tab.
    html = client.get(f"/visit/{new}?pool=all").text
    assert f'href="/visit/{old}" id="nav-next"' in html and "pool=" not in html and "/review" not in html
