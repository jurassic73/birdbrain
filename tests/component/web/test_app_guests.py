"""The Guests page and the public mirror's counting, through the web app: page views and the click beacon count on
the mirror only, the owner's switch and bots are left out, the counts feed opens only to its token, and the page
itself is the admin's on the Pi and a 404 on the mirror."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from birdbrain import audience
from birdbrain.web import app as webapp, viewer

BROWSER = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15"
VISITOR = {"cf-connecting-ip": "203.0.113.9", "cf-ipcountry": "NZ", "user-agent": BROWSER}
TEMPLATES = Path(webapp.__file__).parent / "templates"
STATIC = Path(webapp.__file__).parent / "static"


def counted() -> dict[str, int]:
    audience.flush()
    out: dict[str, int] = {}
    for _, metric, n in audience.rows_since(0):
        out[metric] = out.get(metric, 0) + n
    return out


@pytest.fixture
def adb(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIENCE_DB", str(tmp_path / "state" / "audience.db"))
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(tmp_path / "no-gate.pw"))
    return tmp_path / "state" / "audience.db"


@pytest.fixture
def mirror(conn, adb, monkeypatch):
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    with TestClient(webapp.app) as c:
        yield c


def test_a_page_view_on_the_mirror_counts_the_page_visitor_and_country(mirror):
    r = mirror.get("/", headers=VISITOR)
    assert r.status_code == 200 and 'name="bb-page" content="home"' in r.text and "/static/track.js" in r.text
    assert audience.SEEN_COOKIE in r.headers["set-cookie"]
    mirror.get("/about", headers=VISITOR)
    assert counted() == {"page:home": 1, "page_people:home": 1, "page:about": 1, "page_people:about": 1,
                         "visitors": 1, "country:NZ": 1}
    mirror.get("/stats", headers=VISITOR)  # the cookie came back: a returning browser, but already counted today
    assert counted()["visitors"] == 1


def test_swaps_prefetches_bots_and_local_checks_dont_count(mirror):
    mirror.get("/", headers={**VISITOR, "hx-request": "true"})
    mirror.get("/", headers={**VISITOR, "sec-fetch-dest": "empty"})
    mirror.get("/visit/1", headers={**VISITOR, "sec-purpose": "prefetch;prerender"})
    mirror.get("/", headers={**VISITOR, "user-agent": "Googlebot/2.1"})
    mirror.get("/", headers={"user-agent": BROWSER})  # no CF-Connecting-IP: a check on the server itself
    mirror.get("/api/latest-visit", headers=VISITOR)
    mirror.get("/static/style.css", headers=VISITOR)
    assert counted() == {}


def test_the_owners_switch(mirror):
    r = mirror.get("/?count=no", headers=VISITOR)
    assert re.search(audience.NOCOUNT_COOKIE + r"=1;.*Max-Age", r.headers["set-cookie"])
    mirror.get("/live", headers=VISITOR)
    mirror.post("/api/track?to=theme", headers=VISITOR)
    assert counted() == {}
    r = mirror.get("/?count=yes", headers=VISITOR)
    assert audience.NOCOUNT_COOKIE + '=""' in r.headers["set-cookie"] or "Max-Age=0" in r.headers["set-cookie"]
    assert counted()["page:home"] == 1


def test_both_cookies_survive_under_a_base_path(mirror, monkeypatch):
    """The base_path middleware rewrites the page; a response setting two cookies must still set both."""
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    mirror.cookies.set(audience.NOCOUNT_COOKIE, "1")
    r = mirror.get("/?count=yes", headers=VISITOR)
    cookies = r.headers.get_list("set-cookie")
    assert len(cookies) == 2 and all("Path=/birdbrain/" in c for c in cookies)
    assert int(r.headers["content-length"]) == len(r.content)


def test_the_beacon_counts_allowlisted_clicks_and_prerendered_views(mirror):
    r = mirror.post("/api/track?to=palette", headers=VISITOR)
    assert r.status_code == 204 and r.headers["cache-control"] == "no-store" and not r.content
    mirror.post("/api/track?to=palette", headers=VISITOR)
    assert mirror.post("/api/track?to=drop-tables", headers=VISITOR).status_code == 204
    mirror.post("/api/track?view=visit", headers=VISITOR)
    mirror.post("/api/track?view=settings", headers=VISITOR)
    assert counted() == {"click:palette": 2, "click_people:palette": 1, "page:visit": 1, "page_people:visit": 1,
                         "visitors": 1, "country:NZ": 1}


def test_the_beacon_refuses_other_sites(mirror):
    r = mirror.post("/api/track?to=theme", headers={**VISITOR, "sec-fetch-site": "cross-site"})
    assert r.status_code == 403 and counted() == {}


def test_the_pi_never_counts(conn, adb):
    with TestClient(webapp.app) as c:  # no password: everyone is admin, so the POST gets through to the route
        r = c.get("/", headers=VISITOR)
        assert "track.js" not in r.text and "bb-page" not in r.text and audience.SEEN_COOKIE not in r.headers.get("set-cookie", "")
        assert c.post("/api/track?to=theme", headers=VISITOR).status_code == 204
    assert audience.counter.take() == [] and not adb.exists()


def test_guests_page_is_admin_only_on_the_pi(conn, adb, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    with TestClient(webapp.app) as c:
        r = c.get(audience.PATH, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login?next=%2Fguests"
        assert f'href="{audience.PATH}"' not in c.get("/").text  # no nav tab for visitors
        c.post("/login", data={"password": "s3cret!", "next": "/"})
        page = c.get(audience.PATH)
        assert page.status_code == 200 and f'href="{audience.PATH}" aria-current="page">{audience.NAME}<' in page.text


def test_guests_page_and_feed_are_404s_on_the_mirror(mirror):
    nowhere = mirror.get("/no-such-page")
    for path in (audience.PATH, audience.FEED_PATH):
        r = mirror.get(path, headers=VISITOR)
        assert (r.status_code, r.content) == (404, nowhere.content), path


def test_feed_needs_the_token(mirror, monkeypatch):
    mirror.get("/", headers=VISITOR)
    nowhere = mirror.get("/no-such-page")
    monkeypatch.setenv("AUDIENCE_TOKEN", "the-right-token")
    for headers in ({}, {audience.TOKEN_HEADER: "wrong"}, {audience.TOKEN_HEADER: ""}):
        r = mirror.get(audience.FEED_PATH, headers=headers)
        assert (r.status_code, r.content) == (404, nowhere.content), headers
    r = mirror.get(audience.FEED_PATH + "?since=0", headers={audience.TOKEN_HEADER: "the-right-token"})
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert {m for _, m, _ in r.json()["rows"]} >= {"page:home", "visitors", "country:NZ"}


def test_feed_token_gets_past_the_guest_gate_and_nothing_else_does(mirror, tmp_path, monkeypatch):
    pw = tmp_path / "viewer.pw"
    pw.write_text(viewer.make_hash("test-password-123") + "\n")
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(pw))
    monkeypatch.setenv("AUDIENCE_TOKEN", "the-right-token")
    assert mirror.get(audience.FEED_PATH, headers={audience.TOKEN_HEADER: "the-right-token"}).status_code == 200
    r = mirror.get(audience.FEED_PATH, headers={audience.TOKEN_HEADER: "wrong"}, follow_redirects=False)
    assert r.status_code == 303 and "/guest-login" in r.headers["location"]  # like any other path


def test_feed_is_a_404_on_the_pi(conn, adb, monkeypatch):
    monkeypatch.setenv("AUDIENCE_TOKEN", "the-right-token")
    with TestClient(webapp.app) as c:
        assert c.get(audience.FEED_PATH, headers={audience.TOKEN_HEADER: "the-right-token"}).status_code == 404


def test_guests_page_renders_empty_then_with_counts(conn, adb):
    with TestClient(webapp.app) as c:
        empty = c.get(audience.PATH)
        assert empty.status_code == 200 and "No counts yet" in empty.text and "AUDIENCE_TOKEN" in empty.text
        assert "No IP addresses stored" in empty.text
        slot = int(webapp.clock.now().timestamp()) // 300 * 300
        audience.merge([(slot, "visitors", 4), (slot, "page:home", 7), (slot, "page_people:home", 4),
                        (slot, "country:NZ", 3), (slot, "country:XX", 1), (slot, "click:theme", 2),
                        (slot, "click_people:theme", 2)])
        for period in ("today", "7d", "30d", "year", "bogus"):
            page = c.get(f"{audience.PATH}?period={period}")
            assert page.status_code == 200, period
            text = page.text
            assert "Visits (home)" in text and "Light/dark toggle" in text and "World map" in text
            assert "\U0001F1F3\U0001F1FF" in text and "Unknown" in text and 'lineChart("g-visitors"' in text
            assert "\u2014" not in text  # no em dashes
        assert '7 days</a>' in c.get(audience.PATH).text and 'aria-current="page">7 days' in c.get(audience.PATH).text


def test_every_tracked_control_is_on_the_allowlist():
    """data-track names in the templates and bbTrack("...") calls in the scripts must be audience.CONTROLS names, or
    their clicks would never count; and the controls named there must all exist somewhere."""
    used = set()
    for f in [*TEMPLATES.glob("*.html"), *STATIC.glob("*.js")]:
        text = f.read_text()
        used |= set(re.findall(r'data-track="([a-z_]+)"', text))
        used |= set(re.findall(r'bbTrack\?*\.?\("([a-z_]+)"\)', text))
        for expr in re.findall(r'data-track="\{\{([^}]*)\}\}"', text):  # the zoom macro picks one of two
            used |= set(re.findall(r"'([a-z_]+)'", expr))
    used.discard("name")
    assert used - set(audience.CONTROLS) == set(), "names not on audience.CONTROLS"
    assert set(audience.CONTROLS) - used == set(), "audience.CONTROLS names that nothing sends"
