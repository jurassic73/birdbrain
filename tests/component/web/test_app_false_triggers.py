"""Component tests of the web app through TestClient on a temp database: false triggers."""

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
    assert f'href="/visit/{ft_new}?ft=1"' in client.get("/false-triggers").text
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
    assert r.status_code == 303 and r.headers["location"] == "/false-triggers?cleared=1"
    html = client.get("/false-triggers?cleared=1").text
    assert "Deleted 1 photo. The Pi keeps what it learned from them." in html
    assert "without photos" not in html and f'href="/visit/{vid}?ft=1"' not in html
    assert db.count_visits(conn, false_triggers=True) == 3  # the record stays
    page = client.get(f"/visit/{vid}").text
    assert "Photo deleted to save space" in page and "/images/" + db.get_visit(conn, vid)["image_path"] not in page


def test_false_triggers_come_in_pages(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "FT_PAGE", 2)
    for _ in range(3):
        add_visit(visitor="none", species=None)
    html = client.get("/false-triggers").text
    assert html.count('class="card"') == 2 and "/false-triggers/more?before=" in html
    more = client.get(more_link(html)).text
    assert more.count('class="card"') == 1 and "?ft=1" in more and "load-more" not in more


def test_false_trigger_cards_refresh_after_the_pop_up(client, add_visit, conn):
    keep = add_visit(visitor="none", species=None)
    real = add_visit(visitor="none", species=None, corrected_visitor="bird", corrected_species="Bushtit")
    db.confirm_visits(conn, [keep])
    r = client.get(f"/false-triggers/cards?ids={keep},{real},abc")
    assert f'data-id="{keep}"' in r.text and f'data-id="{real}"' not in r.text
    assert "Confirmed No Animal" in r.text
    assert r.headers["X-Unchecked"] == "0"


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
