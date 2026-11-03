import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

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
    # Requests arrive from the local proxy, so the Cloudflare header names the visitor (as on the droplet).
    with TestClient(webapp.app, client=("127.0.0.1", 50000)) as c:
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


def test_many_addresses_guessing_slow_everyone_down_but_lock_nobody_out(gated, monkeypatch):
    slept = []
    monkeypatch.setattr(webapp.time, "sleep", slept.append)
    for n in range(viewer.ALL_FAILS):
        gated.post("/guest-login", data={"password": "wrong"}, headers=ip(100 + n % 100))
    assert not slept and viewer.slow_down() == viewer.SLOW_DELAY
    # Every try now waits a little, wrong or right, and a fresh address with the right password still gets in.
    assert gated.post("/guest-login", data={"password": "wrong"}, headers=ip(251)).status_code == 401
    assert slept == [viewer.SLOW_DELAY]
    r = gated.post("/guest-login", data={"password": "test-password-123"}, headers=ip(250), follow_redirects=False)
    assert r.status_code == 303 and slept == [viewer.SLOW_DELAY] * 2
    # Whoever is already signed in isn't held up at all (the cookie from the login above).
    assert gated.get("/", follow_redirects=False).status_code == 200
    assert gated.post("/guest-login", data={"password": "whatever"}, headers=ip(252),
                      follow_redirects=False).status_code == 303
    assert slept == [viewer.SLOW_DELAY] * 2


def test_share_link_is_not_slowed_by_everyone_else_guessing(gated, tmp_path, monkeypatch):
    monkeypatch.setenv("VIEWER_INVITE_FILE", str(tmp_path / "viewer.invite"))
    key = viewer.make_link("https://somegeek.com/birdbrain").split("#k=")[1]
    slept = []
    monkeypatch.setattr(webapp.time, "sleep", slept.append)
    for n in range(viewer.ALL_FAILS):
        viewer.failed(f"198.51.100.{n}")
    r = gated.post("/guest-login", data={"key": key, "next": "/"}, headers=ip(60), follow_redirects=False)
    assert r.status_code == 303 and viewer.COOKIE in r.headers["set-cookie"] and not slept


def test_a_look_at_the_login_page_never_adds_a_throttle_entry(gated):
    for n in range(50):
        assert gated.get("/guest-login", headers=ip(n)).status_code == 200
        gated.get("/", headers=ip(n), follow_redirects=False)
    assert viewer._fails == {}
    assert viewer.locked("203.0.113.1") == 0 and viewer._fails == {}


def test_old_tries_are_forgotten_and_the_table_stays_small(monkeypatch):
    viewer._fails.clear()
    monkeypatch.setattr(viewer, "MAX_TRACKED", 5)
    now = time.time()
    viewer._fails["stale"] = [now - viewer.LOCKOUT - 1]
    viewer._fails["mixed"] = [now - viewer.LOCKOUT - 1, now - 1]
    viewer.failed("fresh")
    assert "stale" not in viewer._fails and viewer._fails["mixed"] == [pytest.approx(now - 1)]
    assert viewer.locked("stale") == 0
    for n in range(20):
        viewer.failed(f"10.0.0.{n}")
    assert len(viewer._fails) <= 6 and "*" in viewer._fails  # MAX_TRACKED addresses plus everyone's count
    assert len(viewer._fails["*"]) <= viewer.ALL_FAILS
    for _ in range(10):
        viewer.failed("10.0.0.1")
    assert len(viewer._fails["10.0.0.1"]) == viewer.PER_IP_FAILS and viewer.locked("10.0.0.1") > 0
    viewer._fails.clear()


def test_cloudflare_header_is_only_believed_from_the_local_proxy(tmp_path, monkeypatch, conn):
    pw = tmp_path / "viewer.pw"
    pw.write_text(viewer.make_hash("test-password-123") + "\n")
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(pw))
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    viewer._fails.clear()
    viewer._cache.clear()
    try:
        with TestClient(webapp.app, client=("198.51.100.7", 4000)) as direct:
            # A visitor sending the header can't make their tries count against someone else.
            for n in range(viewer.PER_IP_FAILS):
                direct.post("/guest-login", data={"password": "wrong"}, headers=ip(n))
            assert set(viewer._fails) == {"198.51.100.7", "*"}
            assert direct.post("/guest-login", data={"password": "test-password-123"}, headers=ip(99),
                               follow_redirects=False).status_code == 429
        monkeypatch.setenv("VIEWER_TRUSTED_PROXIES", "198.51.100.7, 10.0.0.1")
        assert "198.51.100.7" in viewer.trusted_proxies() and "10.0.0.1" in viewer.trusted_proxies()
        with TestClient(webapp.app, client=("198.51.100.7", 4000)) as proxied:
            assert proxied.post("/guest-login", data={"password": "test-password-123"}, headers=ip(99),
                                follow_redirects=False).status_code == 303
    finally:
        viewer._fails.clear()
        viewer._cache.clear()


def test_a_bad_cookie_means_not_signed_in(gated):
    def request_with(cookie: bytes) -> Request:
        return Request({"type": "http", "method": "GET", "path": "/", "headers": [(b"cookie", cookie)],
                        "query_string": b"", "client": ("127.0.0.1", 1)})

    far = str(int(time.time()) + 100)
    for cookie in (b"bb_guest=" + far.encode() + b".\xe9\xff", b"bb_guest=\xb2.abc", b"bb_guest=" + far.encode(),
                   b"bb_guest=.", b"bb_guest=abc.def", "bb_guest=١٢٣.abc".encode("utf-8")):
        assert viewer.signed_in(request_with(cookie)) is False, cookie
    for value in ("123", ".", "abc.def", f"{far}.notahexdigest", f"{far}.", "%E9%FF.%E9"):
        r = gated.get("/", headers={"cookie": f"bb_guest={value}"}, follow_redirects=False)
        assert r.status_code == 303, value
    assert viewer.signed_in(request_with(b"bb_guest=" + viewer.make_cookie().encode())) is True


def test_new_password_signs_everyone_out(gated, tmp_path):
    gated.post("/guest-login", data={"password": "test-password-123"}, headers=ip(6))
    assert gated.get("/", follow_redirects=False).status_code == 200
    pw = tmp_path / "viewer.pw"
    pw.write_text(viewer.make_hash("somethingElse9") + "\n")
    os.utime(pw, (time.time() + 5, time.time() + 5))
    assert gated.get("/", follow_redirects=False).status_code == 303


def test_no_password_file_means_no_gate(conn, monkeypatch, tmp_path):
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(tmp_path / "missing"))
    viewer._cache.clear()
    assert viewer._read(tmp_path / "missing") == "" and not viewer.enabled()
    with TestClient(webapp.app) as c:
        assert c.get("/", follow_redirects=False).status_code == 200


def test_an_unreadable_password_file_shuts_the_site(gated, tmp_path, monkeypatch, caplog):
    pw = tmp_path / "viewer.pw"
    real_stat = Path.stat

    def denied(self, *args, **kwargs):
        if self == pw:
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *args, **kwargs)

    # Signed in first: even a valid cookie doesn't get past a gate that can't be checked.
    gated.post("/guest-login", data={"password": "test-password-123"}, headers=ip(7))
    monkeypatch.setattr(Path, "stat", denied)
    viewer._cache.clear()
    with pytest.raises(viewer.Unreadable):
        viewer.enabled()
    for path in ("/", "/guest-login", "/manifest.webmanifest", "/static/style.css"):
        r = gated.get(path, follow_redirects=False)
        assert r.status_code == 503 and "closed for a moment" in r.text, path
    assert gated.post("/guest-login", data={"password": "test-password-123"}, headers=ip(8)).status_code == 503
    assert "can't be read" in caplog.text
    # A directory in the file's place is just as unreadable; a missing file is still "no gate".
    monkeypatch.setattr(Path, "stat", real_stat)
    viewer._cache.clear()
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(tmp_path))
    assert gated.get("/", follow_redirects=False).status_code == 503


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


def test_viewer_files_never_resolve_under_etc_in_tests():
    """Guard: the autouse fixture keeps the gate off the host's real /etc/birdbrain files."""
    for p in (viewer.password_file(), viewer.invite_file()):
        assert not str(p).startswith("/etc"), p
