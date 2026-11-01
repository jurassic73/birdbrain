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
