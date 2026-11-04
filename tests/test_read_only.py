import pytest
from fastapi.testclient import TestClient

from conftest import FROZEN_NOW
from birdbrain import db, images, species_info
from birdbrain.web import app as webapp
from birdbrain.web import auth


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def locked(monkeypatch):
    """Read-only mode: an admin password is set."""
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()


def test_visitors_browse_but_change_nothing(client, add_visit, conn, locked):
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    page = client.get(f"/visit/{vid}")
    assert page.status_code == 200 and "Dark-eyed Junco" in page.text
    assert "/correct" not in page.text and "Delete visit" not in page.text and 'href="/settings"' not in page.text
    assert 'href="/login' in page.text
    for path in (f"/visit/{vid}/correct", f"/visit/{vid}/delete", f"/visit/{vid}/check-circle/pi",
                 f"/visit/{vid}/check-circle/haiku", "/notes/refine", "/settings/alerts", "/settings/mirror",
                 "/settings/zones/manual"):
        assert client.post(path).status_code == 403
    assert db.get_visit(conn, vid) is not None
    for path in ("/settings", "/notes", "/review", "/false-triggers", "/false-triggers/cards?after=0",
                 "/settings/camera", "/lessons"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/login")
    for path in ("/", "/live", "/species", "/stats", "/about"):
        assert client.get(path).status_code == 200


def test_visitors_species_card_never_fetches(client, locked, monkeypatch):
    seen = []
    monkeypatch.setattr(species_info, "get", lambda conn, name, fetch=True: seen.append(fetch))
    client.get("/species-info/Bushtit")
    assert seen == [False]


def test_admin_login_and_logout(client, add_visit, locked):
    vid = add_visit(visitor="bird", species="Bushtit")
    assert client.post("/login", data={"password": "wrong"}).status_code == 401
    r = client.post("/login", data={"password": "s3cret!", "next": "//evil.example"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/settings").status_code == 200
    assert f'hx-post="/visit/{vid}/correct"' in client.get(f"/visit/{vid}").text
    client.post("/logout")
    assert client.get("/settings", follow_redirects=False).status_code == 303


def test_changing_the_password_signs_everyone_out(client, locked, monkeypatch):
    client.post("/login", data={"password": "s3cret!"})
    monkeypatch.setenv("ADMIN_PASSWORD", "new-one")
    assert client.get("/settings", follow_redirects=False).status_code == 303


def test_login_locks_out_after_repeated_wrong_passwords(client, locked):
    for _ in range(5):
        client.post("/login", data={"password": "nope"})
    assert client.post("/login", data={"password": "s3cret!"}).status_code == 429


def test_no_admin_password_means_everyone_is_admin(client):
    assert client.get("/settings").status_code == 200
    assert "Log out" not in client.get("/").text and 'href="/login' not in client.get("/").text


def test_public_mirror_has_no_admin_and_never_touches_camera(client, add_visit, conn, monkeypatch):
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")  # even with a password set, nobody can log in
    async def no_camera():
        raise AssertionError("public mirror contacted the camera")
    monkeypatch.setattr(webapp, "grab_live_frame", no_camera)
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    page = client.get(f"/visit/{vid}")
    assert page.status_code == 200 and "Dark-eyed Junco" in page.text
    assert 'href="/login' not in page.text and "Log out" not in page.text and "/correct" not in page.text
    assert client.get("/login").status_code == 404
    assert client.post("/login", data={"password": "s3cret!"}).status_code == 404
    assert client.post(f"/visit/{vid}/delete").status_code == 404
    for path in ("/settings", "/notes", "/review", "/false-triggers", "/settings/camera", "/lessons"):
        assert client.get(path, follow_redirects=False).status_code == 404
    for path in ("/", "/live", "/species", "/stats", "/about"):
        assert client.get(path).status_code == 200
    assert client.get("/live.jpg").status_code == 503  # no synced frame; must not reach the camera either way
    assert db.get_visit(conn, vid) is not None


def test_base_path_prefixes_links_redirects_and_fetch(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    page = client.get("/").text
    assert f'href="/birdbrain/visit/{vid}"' in page and 'href="/visit/' not in page
    assert 'src="/birdbrain/static/' in page and 'src="/static/' not in page
    assert "window.BB_BASE=\"/birdbrain\"" in page
    r = client.post("/logout", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/birdbrain/"


def test_mirror_today_follows_site_timezone(client, add_visit, conn, monkeypatch):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from birdbrain import clock
    # A zone whose date differs from this machine's for most of the day, so "today" visibly moves.
    def in_zone(name):  # the frozen "now" as the wall clock in that zone
        return FROZEN_NOW.astimezone().astimezone(ZoneInfo(name)).replace(tzinfo=None)
    zone = "Pacific/Kiritimati" if in_zone("Pacific/Kiritimati").date() != FROZEN_NOW.date() else "Etc/GMT+12"
    there = in_zone(zone)
    assert there.date() != FROZEN_NOW.date()
    try:
        db.set_value(conn, clock.KEY, zone)
        clock.set_zone(None); clock._zone["checked"] = 0.0  # make the next request read the setting
        assert client.get("/").status_code == 200
        assert clock.today() == there.date()
        db.set_value(conn, clock.KEY, "")
        clock._zone["checked"] = 0.0
        assert client.get("/").status_code == 200
        assert clock.today() == FROZEN_NOW.date()
    finally:
        clock.set_zone(None); clock._zone["checked"] = 0.0


def test_somegeek_home_logo_only_on_public_mirror(client, monkeypatch):
    assert "somegeek.com" not in client.get("/").text
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    page = client.get("/").text
    assert 'href="https://somegeek.com/"' in page and 'aria-label="somegeek.com home"' in page
    assert 'src="/birdbrain/static/somegeek-logo.png' in page


def test_latest_visit_api_waiting_counts_only_for_admin(client, add_visit, locked, monkeypatch):
    add_visit(needs_review=1)
    add_visit(visitor="none", species=None)
    add_visit(visitor="none", species=None)
    zero = {"review": 0, "false": 0}
    assert client.get("/api/latest-visit").json()["waiting"] == zero  # visitor
    client.post("/login", data={"password": "s3cret!"})
    assert client.get("/api/latest-visit").json()["waiting"] == {"review": 1, "false": 2}
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")  # the public mirror never shows counts
    assert client.get("/api/latest-visit").json()["waiting"] == zero


def test_visitors_cannot_draw_circles(client, add_visit, locked):
    vid = add_visit(visitor="bird", species="Bushtit")  # a false trigger isn't shown to visitors at all
    html = client.get(f"/visit/{vid}").text
    assert "const editing = false;" in html and 'id="draw-note"' not in html


def test_visitors_keyboard_popup_has_no_admin_only_confirm_key(client, add_visit, locked):
    vid = add_visit(visitor="bird", species="Bushtit")
    add_visit(visitor="bird", species="Bushtit")
    html = client.get(f"/visit/{vid}").text
    assert 'id="keys-dialog"' in html and "Newer visit" in html
    assert "Confirm no animal" not in html


def _false_trigger(add_visit, **fields):
    return add_visit(visitor="none", species=None, image_path="2026/11/03/ft.jpg", thumb_path="2026/11/03/ft_t.jpg",
                     **fields)


def test_visitors_get_404_for_a_false_trigger_but_admin_sees_it_with_its_burst(client, add_visit, locked, monkeypatch):
    real = add_visit(visitor="bird", species="Bushtit")
    ft = _false_trigger(add_visit)
    other_ft = _false_trigger(add_visit)
    monkeypatch.setattr(images, "burst_paths", lambda rel: ["2026/11/03/ft_b1.jpg", "2026/11/03/ft_b2.jpg"] if "ft" in rel else [])
    # A visitor: the same 404 as for an id that doesn't exist, however the page is asked for.
    for path in (f"/visit/{ft}", f"/visit/{ft}?ft=1", f"/visit/{ft}?todo=1", f"/visit/{other_ft}", "/visit/99999"):
        assert client.get(path).status_code == 404
    # The ft flag can't turn a real visit's arrows into a walk through false triggers.
    page = client.get(f"/visit/{real}?ft=1")
    assert page.status_code == 200 and f"/visit/{ft}" not in page.text and f"/visit/{other_ft}" not in page.text
    assert "ft_b1.jpg" not in page.text
    # The admin keeps everything: the page, the burst strip and the arrows through the other false triggers.
    assert client.post("/login", data={"password": "s3cret!"}, follow_redirects=False).status_code == 303
    page = client.get(f"/visit/{ft}")
    assert page.status_code == 200 and "ft_b1.jpg" in page.text and "ft_b2.jpg" in page.text
    assert f"/visit/{other_ft}" in page.text
    assert client.get(f"/visit/{real}?ft=1").status_code == 200


def test_no_password_means_everyone_is_admin_and_sees_false_triggers(client, add_visit):
    ft = _false_trigger(add_visit)
    assert client.get(f"/visit/{ft}").status_code == 200


def test_the_public_mirror_has_no_false_trigger_pages(client, add_visit, monkeypatch):
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    ft = _false_trigger(add_visit)
    assert client.get(f"/visit/{ft}").status_code == 404


def test_visitors_get_404_for_a_false_triggers_burst_frames_but_a_real_visits_stay(client, add_visit, locked,
                                                                                  monkeypatch):
    """/images is a plain static folder: a visitor who knows (or guesses) a false trigger's burst frame name gets the
    same 404 as for a missing file. A real visit's burst (a false trigger the owner corrected keeps its burst and is
    shown on its page) stays public, and the admin sees everything."""
    from birdbrain import config
    day = config.IMAGES_DIR / "2026" / "11" / "01"
    day.mkdir(parents=True, exist_ok=True)
    names = ("ft.jpg", "ft_t.jpg", "ft_b1.jpg", "ft_b2.jpg", "ft_b2_t.jpg", "real.jpg", "real_t.jpg", "real_b1.jpg")
    for name in names:
        (day / name).write_bytes(b"jpeg")
    try:
        add_visit(visitor="none", species=None, image_path="2026/11/01/ft.jpg", thumb_path="2026/11/01/ft_t.jpg")
        add_visit(visitor="none", species=None, corrected_visitor="bird", corrected_species="Bushtit",
                  image_path="2026/11/01/real.jpg", thumb_path="2026/11/01/real_t.jpg")
        hidden = ("/images/2026/11/01/ft_b1.jpg", "/images/2026/11/01/ft_b2.jpg", "/images/2026/11/01/ft_b2_t.jpg",
                  "/images//2026/11/01/ft_b1.jpg", "/images/2026/11/01/./ft_b1.jpg", "/images/2026/11/../11/01/ft_b1.jpg")
        shown = ("/images/2026/11/01/real.jpg", "/images/2026/11/01/real_t.jpg", "/images/2026/11/01/real_b1.jpg")
        for url in hidden:
            assert client.get(url).status_code == 404, url
        for url in shown:
            assert client.get(url).status_code == 200, url
        assert client.post("/login", data={"password": "s3cret!"}, follow_redirects=False).status_code == 303
        for url in hidden[:3] + shown:
            assert client.get(url).status_code == 200, url
        client.cookies.clear()
        monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")  # the mirror has no false triggers; its real bursts load
        assert client.get("/images/2026/11/01/real_b1.jpg").status_code == 200
    finally:
        for name in names:
            (day / name).unlink(missing_ok=True)


def test_latest_visit_api_tells_only_the_admin_of_a_false_trigger(client, add_visit, locked, monkeypatch):
    real = add_visit(visitor="bird", species="Bushtit")
    ft = add_visit(visitor="none", species=None)
    v = client.get("/api/latest-visit").json()
    assert (v["id"], v["any_id"]) == (real, real)  # a visitor: the false trigger's id never shows up
    client.post("/login", data={"password": "s3cret!"})
    v = client.get("/api/latest-visit").json()
    assert (v["id"], v["any_id"]) == (real, ft)  # the admin's False triggers page refreshes on it
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    assert client.get("/api/latest-visit").json()["any_id"] == real
