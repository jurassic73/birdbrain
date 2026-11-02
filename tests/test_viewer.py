import pytest
from fastapi.testclient import TestClient

from birdbrain.web import app as webapp
from birdbrain.web import viewer


@pytest.fixture
def gated(tmp_path, monkeypatch, conn):
    pw = tmp_path / "viewer.pw"
    pw.write_text(viewer.make_hash("test-password-123") + "\n")
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(pw))
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    viewer._fails.clear()
    viewer._cache.clear()
    with TestClient(webapp.app) as c:
        yield c
    viewer._fails.clear()
    viewer._cache.clear()


def ip(n):
    return {"cf-connecting-ip": f"203.0.113.{n}"}


def test_nothing_without_the_password(gated, add_visit):
    vid = add_visit(visitor="bird", species="Dark-eyed Junco")
    for path in ("/", f"/visit/{vid}", "/species", "/stats", "/about", "/live.jpg", "/live/latest.jpg",
                 "/images/anything.jpg", "/species-info/Dark-eyed%20Junco", "/api/latest-visit"):
        r = gated.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/guest-login"), path
    page = gated.get("/", headers=ip(1))
    assert "Friends &amp; family login" in page.text and "Dark-eyed Junco" not in page.text
    assert gated.post(f"/visit/{vid}/delete").status_code == 401


def test_right_password_lets_you_in(gated, add_visit):
    add_visit(visitor="bird", species="Dark-eyed Junco")
    r = gated.post("/guest-login", data={"password": "test-password-123", "next": "/stats"}, headers=ip(2),
                   follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/stats"
    assert "httponly" in r.headers["set-cookie"].lower()
    assert "Dark-eyed Junco" in gated.get("/").text


def test_popup_gets_json(gated):
    js = {"accept": "application/json", **ip(3)}
    assert gated.post("/guest-login", data={"password": "nope"}, headers=js).json()["ok"] is False
    r = gated.post("/guest-login", data={"password": "test-password-123"}, headers=js)
    assert r.json() == {"ok": True, "error": "", "wait": 0} and viewer.COOKIE in r.headers["set-cookie"]


def test_three_wrong_tries_lock_that_address_out(gated):
    for _ in range(2):
        assert gated.post("/guest-login", data={"password": "wrong"}, headers=ip(4)).status_code == 401
    third = gated.post("/guest-login", data={"password": "wrong"}, headers=ip(4))
    assert third.status_code == 429 and "Too many tries" in third.text
    # Even the right password is refused while locked out; another address is unaffected.
    assert gated.post("/guest-login", data={"password": "test-password-123"}, headers=ip(4),
                      follow_redirects=False).status_code == 429
    assert gated.post("/guest-login", data={"password": "test-password-123"}, headers=ip(5),
                      follow_redirects=False).status_code == 303


def test_many_addresses_guessing_lock_everyone(gated):
    for n in range(viewer.ALL_FAILS):
        gated.post("/guest-login", data={"password": "wrong"}, headers=ip(100 + n % 100))
    assert gated.post("/guest-login", data={"password": "test-password-123"}, headers=ip(250),
                      follow_redirects=False).status_code == 429


def test_new_password_signs_everyone_out(gated, tmp_path):
    gated.post("/guest-login", data={"password": "test-password-123"}, headers=ip(6))
    assert gated.get("/", follow_redirects=False).status_code == 200
    import os, time
    pw = tmp_path / "viewer.pw"
    pw.write_text(viewer.make_hash("somethingElse9") + "\n")
    os.utime(pw, (time.time() + 5, time.time() + 5))
    assert gated.get("/", follow_redirects=False).status_code == 303


def test_no_password_file_means_no_gate(conn, monkeypatch, tmp_path):
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(tmp_path / "missing"))
    viewer._cache.clear()
    with TestClient(webapp.app) as c:
        assert c.get("/", follow_redirects=False).status_code == 200


def test_share_link_key_signs_in(gated, tmp_path, monkeypatch):
    monkeypatch.setenv("VIEWER_INVITE_FILE", str(tmp_path / "viewer.invite"))
    link = viewer.make_link("https://somegeek.com/birdbrain")
    assert link.startswith("https://somegeek.com/birdbrain/guest-login#k=")
    key = link.split("#k=")[1]
    assert key not in (tmp_path / "viewer.invite").read_text()  # only its hash is stored
    page = gated.get("/guest-login", headers=ip(20))
    assert 'name="key"' in page.text and "location.hash" in page.text
    assert gated.post("/guest-login", data={"key": "guess"}, headers=ip(21)).status_code == 401
    r = gated.post("/guest-login", data={"key": key, "next": "/"}, headers=ip(22), follow_redirects=False)
    assert r.status_code == 303 and viewer.COOKIE in r.headers["set-cookie"]
    assert gated.get("/", follow_redirects=False).status_code == 200
    # A new link retires the old key and signs everyone out.
    import os, time
    viewer.make_link("https://somegeek.com/birdbrain")
    inv = tmp_path / "viewer.invite"
    os.utime(inv, (time.time() + 5, time.time() + 5))
    assert gated.get("/", follow_redirects=False).status_code == 303
    assert gated.post("/guest-login", data={"key": key}, headers=ip(23)).status_code == 401


def test_password_still_works_without_a_link(gated):
    r = gated.post("/guest-login", data={"password": "test-password-123"}, headers=ip(30), follow_redirects=False)
    assert r.status_code == 303


def test_manifest_is_open_but_only_that_exact_path(gated, monkeypatch):
    # Browsers fetch the manifest without cookies, so the gate must not bounce it (Android home-screen icon).
    r = gated.get("/manifest.webmanifest", follow_redirects=False)
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/manifest+json")
    assert r.json()["name"] == "Birdbrain" and any(i["purpose"] == "maskable" for i in r.json()["icons"])
    assert gated.get("/manifest.webmanifest?v=123", follow_redirects=False).status_code == 200
    # Everything else, including lookalikes, stays behind the gate.
    for path in ("/", "/manifest.webmanifest.bak", "/manifest.webmanifestx", "/manifest.webmanifest/",
                 "/manifest.webmanifest/x", "//manifest.webmanifest", "/x/manifest.webmanifest",
                 "/Manifest.webmanifest", "/manifest.webmanifest%2f.."):
        r = gated.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].startswith("/guest-login"), path
    # Nothing else is opened (the open paths are the login page and the manifest; static stays a prefix).
    assert viewer.OPEN_PATHS == ("/guest-login", "/manifest.webmanifest")
    # Under the public base path the manifest still answers without a cookie, with prefixed URLs.
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    m = gated.get("/manifest.webmanifest", follow_redirects=False).json()
    assert m["start_url"] == "/birdbrain/" and all(i["src"].startswith("/birdbrain/static/") for i in m["icons"])


def test_guest_login_page_links_manifest_and_home_screen_icon(gated, monkeypatch):
    html = gated.get("/guest-login", headers=ip(30)).text
    assert 'rel="manifest" href="/manifest.webmanifest?v=' in html
    assert 'rel="apple-touch-icon" href="/static/apple-touch-icon.png?v=' in html
    assert 'rel="icon" type="image/png" sizes="64x64" href="/static/favicon-64.png' in html
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    html = gated.get("/guest-login", headers=ip(31)).text
    assert 'rel="manifest" href="/birdbrain/manifest.webmanifest?v=' in html
    assert 'rel="apple-touch-icon" href="/birdbrain/static/apple-touch-icon.png?v=' in html
    assert 'href="/birdbrain/static/favicon-64.png' in html


def test_sign_in_under_base_path_returns_to_the_base_home(gated, monkeypatch):
    # The standalone app opens start_url /birdbrain/ with no cookie; signing in must land back inside /birdbrain/.
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    r = gated.get("/", follow_redirects=False)  # what nginx hands us for /birdbrain/
    assert r.status_code == 303 and r.headers["location"] == "/birdbrain/guest-login?next=%2F"
    page = gated.get("/guest-login?next=%2F", headers=ip(32)).text
    assert 'action="/birdbrain/guest-login"' in page and 'name="next" value="/"' in page
    r = gated.post("/guest-login", data={"password": "test-password-123", "next": "/"}, headers=ip(33),
                   follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/birdbrain/"
    assert "Path=/birdbrain/" in r.headers["set-cookie"]
