"""The LAN site's guards: every route classified (visitor, admin, public mirror), cross-site POSTs refused,
the admin cookie's signing and parsing, and species cards built only for animals the visits know."""
import asyncio
import json
import os
import stat
import time

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.routing import Route

from birdbrain import db, species_info
from birdbrain.web import app as webapp
from birdbrain.web import auth, viewer


@pytest.fixture
def client(conn, monkeypatch):
    async def no_camera():
        return b""  # /live.jpg answers 503, nothing is contacted
    monkeypatch.setattr(webapp, "grab_live_frame", no_camera)
    with TestClient(webapp.app) as c:
        yield c


@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    auth._fails.clear()


@pytest.fixture
def mirror(monkeypatch):
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")


# Every GET a visitor may make, by route template. A new route that is neither here nor under auth.ADMIN_PATHS
# fails test_every_route_is_classified: decide where it belongs when you add it.
PUBLIC_GETS = {
    "/", "/about", "/api/latest-visit", "/api/photo-reminder", "/favicon.ico", "/guest-login", "/live.jpg",
    "/live/latest.jpg", "/manifest.webmanifest", "/scoreboard", "/species", "/species-info/{name}",
    "/species-photo", "/species-search", "/species/{name}", "/species/{name}/cover-compare", "/stats",
    "/visit/{visit_id}", "/visits", "/visits/archive-more", "/visits/more", "/weather",
}
LOGIN_ROUTES = {"/login", "/logout", "/guest-login"}
FILL = {"visit_id": "1", "name": "x", "source": "pi", "state": "x", "change_id": "1", "cal_id": "1"}


def _routes() -> list[tuple[str, str, str]]:
    """(method, template, concrete path) for every app route, e.g. ('GET', '/calibration/{name}', '/calibration/x')."""
    out = []
    for r in webapp.app.routes:
        if not isinstance(r, Route):
            continue  # the /static and /images mounts
        path = r.path
        for k, v in FILL.items():
            path = path.replace("{" + k + "}", v)
        assert "{" not in path, f"add a FILL value for {r.path}"
        for m in sorted(r.methods or []):
            if m != "HEAD":
                out.append((m, r.path, path))
    return out


ROUTES = _routes()
IDS = [f"{m} {p}" for m, _, p in ROUTES]


def test_route_list_is_real():
    assert ("GET", "/calibration/{name}", "/calibration/x") in ROUTES
    assert ("POST", "/settings/camera/reboot", "/settings/camera/reboot") in ROUTES


@pytest.mark.parametrize("method,template,path", ROUTES, ids=IDS)
def test_every_route_is_classified(client, locked, method, template, path):
    """With a password set and no cookie: admin pages send you to /login, every change is refused, and
    anything else a visitor may fetch is on the allowlist above."""
    if method == "GET":
        r = client.get(path, follow_redirects=False)
        if auth.admin_only(template):
            assert r.status_code == 303 and r.headers["location"].startswith("/login?"), path
            assert template not in PUBLIC_GETS
        elif template == "/login":
            assert r.status_code == 200
        else:
            assert template in PUBLIC_GETS, f"new route {template}: public, or add it to auth.ADMIN_PATHS?"
            assert r.status_code != 500 and not r.headers.get("location", "").startswith("/login"), path
    elif template not in LOGIN_ROUTES:
        r = client.request(method, path)
        assert r.status_code == 403 and "log in" in r.text, path


@pytest.mark.parametrize("method,template,path", ROUTES, ids=IDS)
def test_every_route_on_the_public_mirror(client, mirror, method, template, path):
    """BIRDBRAIN_PUBLIC=1: the admin pages and the login don't exist, nothing can be changed."""
    r = client.request(method, path, follow_redirects=False)
    if method != "GET":
        if template != "/guest-login":
            assert r.status_code == 403, path
    elif auth.admin_only(template) or template in ("/login", "/species-search"):
        assert r.status_code == 404, path
    else:
        assert r.status_code != 500 and not r.headers.get("location", "").startswith("/login"), path


# --- Cross-site requests -------------------------------------------------------------------------------------

@pytest.fixture(params=["open", "locked"])
def admin_client(request, client, add_visit):
    """A client that may change things: with no password at all, or logged in with one."""
    if request.param == "locked":
        request.getfixturevalue("locked")
        client.post("/login", data={"password": "s3cret!"})
    return client


def test_cross_site_posts_are_refused(admin_client, add_visit, conn):
    vid = add_visit()
    for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}, {"Sec-Fetch-Site": "none"},
                    {"Origin": "http://evil.example"}, {"Origin": "http://testserver:8001"}, {"Origin": "null"},
                    {"Origin": "http://evil.example", "Sec-Fetch-Site": "cross-site"}):
        r = admin_client.post(f"/visit/{vid}/delete", headers=headers)
        assert r.status_code == 403 and "Cross-site" in r.text, headers
        r = admin_client.post("/login", data={"password": "s3cret!"}, headers=headers)
        assert r.status_code == 403, headers
    assert db.get_visit(conn, vid) is not None


def test_same_origin_and_headerless_posts_work(admin_client, add_visit, conn):
    for headers in ({}, {"Sec-Fetch-Site": "same-origin"}, {"Origin": "http://testserver"},
                    {"Origin": "http://TestServer:80"}, {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"},
                    {"Origin": "http://evil.example", "Sec-Fetch-Site": "same-origin"}):  # the browser's word wins
        vid = add_visit()
        assert admin_client.post(f"/visit/{vid}/reviewed", headers=headers).status_code == 200, headers
        assert db.get_visit(conn, vid)["confirmed_at"], headers


def test_cross_site_check_runs_before_everything_else(client, add_visit, locked):
    """Even a visitor's cross-site POST, and the public mirror's guest login, say why they were refused."""
    vid = add_visit()
    assert "Cross-site" in client.post(f"/visit/{vid}/delete", headers={"Sec-Fetch-Site": "cross-site"}).text
    assert "Cross-site" in client.post("/guest-login", data={"password": "x"}, headers={"Origin": "http://evil.example"}).text
    assert client.get("/", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200  # GETs are never blocked


def test_same_origin_under_a_base_path(client, add_visit, monkeypatch):
    monkeypatch.setattr(webapp, "BASE", "/birdbrain")
    vid = add_visit()
    ok = {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"}
    assert client.post(f"/visit/{vid}/reviewed", headers=ok).status_code == 200
    r = client.post("/logout", headers=ok, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/birdbrain/"
    assert client.post(f"/visit/{vid}/delete", headers={"Origin": "http://somegeek.com"}).status_code == 403


def test_https_origin_behind_the_proxy():
    """nginx passes Host without the port; the browser's Origin for an https page has none either."""
    assert auth._host_port("somegeek.com", "https") == auth._host_port("somegeek.com:443", "https") == "somegeek.com"
    assert auth._host_port("192.168.1.205:8000", "http") == "192.168.1.205:8000"


# --- The admin cookie ----------------------------------------------------------------------------------------

def _status(path: str, cookie: bytes) -> int:
    """The app's answer to a GET carrying a raw Cookie header (bytes, so it can hold what no client would send)."""
    scope = {"type": "http", "http_version": "1.1", "method": "GET", "scheme": "http", "path": path,
             "raw_path": path.encode(), "query_string": b"", "root_path": "", "client": ("127.0.0.1", 1),
             "server": ("testserver", 80), "headers": [(b"host", b"testserver"), (b"cookie", cookie)]}
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(webapp.app(scope, receive, send))
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_bad_cookies_are_not_admin_and_never_crash(client, locked):
    now = int(time.time())
    bad = ["", "garbage", "123", "123.", ".abc", f"{now + 100}.zz", f"{now + 100}.{'0' * 64}",  # wrong signature
           f"{now - 10}.{auth._sign(now - 10)}",  # expired
           f"{now + 100}.{auth._sign(now + 100)[:-1]}", "1e5." + "a" * 64, f"{now + 100}.{'a' * 64}.extra"]
    for value in bad:
        client.cookies.set(auth.COOKIE, value)
        assert client.get("/settings", follow_redirects=False).status_code == 303, value
    client.cookies.clear()
    for raw in [b"bb_admin=123.\xff\xfe", b"bb_admin=\xc3\xbf", f"bb_admin={now + 100}.\xff".encode("latin-1"),
                "bb_admin=123.ÿ€".encode("utf-8"), b"bb_admin=" + b"\xff" * 80, b"bb_admin=\xff; bb_guest=\xfe"]:
        assert _status("/settings", raw) == 303, raw
        assert _status("/", raw) == 200, raw


def test_non_ascii_cookie_values_are_simply_not_signed_in(locked, tmp_path, monkeypatch):
    def request(cookie: str) -> Request:
        return Request({"type": "http", "method": "GET", "path": "/", "query_string": b"",
                        "headers": [(b"cookie", cookie.encode("latin-1"))]})
    future = int(time.time()) + 100
    assert auth.is_admin(request(f"bb_admin={future}.ÿ")) is False
    assert auth.is_admin(request("bb_admin=ÿ")) is False
    pw = tmp_path / "viewer.pw"
    pw.write_text(viewer.make_hash("test-password-123") + "\n")
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(pw))
    viewer._cache.clear()
    try:
        assert viewer.signed_in(request(f"bb_guest={future}.ÿ")) is False
        assert viewer.signed_in(request(f"bb_guest={future}.{'0' * 64}")) is False
    finally:
        viewer._cache.clear()


def test_valid_cookie_works_until_the_password_changes(client, locked, monkeypatch):
    client.cookies.set(auth.COOKIE, auth.make_cookie())
    assert client.get("/settings").status_code == 200
    monkeypatch.setenv("ADMIN_PASSWORD", "another")
    assert client.get("/settings", follow_redirects=False).status_code == 303
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")
    assert client.get("/settings").status_code == 200


def test_cookie_key_comes_from_a_private_random_secret(client, locked):
    path = auth.config.DATA_DIR / auth.SECRET_FILE
    cookie = auth.make_cookie()
    assert path.exists() and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(bytes.fromhex(path.read_text().strip())) == 32  # 64 hex characters of os.urandom
    # Without the secret, the key would follow from the password alone (sha256 of a known prefix + password).
    import hashlib
    assert auth._key() != hashlib.sha256(b"birdbrain-admin:s3cret!").digest()
    client.cookies.set(auth.COOKIE, cookie)
    assert client.get("/settings").status_code == 200
    # A new secret (the file removed, e.g. a fresh data dir) signs everyone out, like a new password.
    saved = path.read_text()
    try:
        path.unlink()
        auth._secrets.clear()
        assert client.get("/settings", follow_redirects=False).status_code == 303
        assert path.exists() and path.read_text() != saved
    finally:
        path.write_text(saved)
        os.chmod(path, 0o600)
        auth._secrets.clear()


# --- Species cards only for animals the visits know ----------------------------------------------------------

@pytest.fixture
def offline(monkeypatch):
    """Any outside lookup is a test failure; build() returns a card worth caching."""
    built = []

    def build(name):
        built.append(name)
        return {"name": name, "inat_id": 1, "links": []}

    def boom(*a, **k):
        raise AssertionError("no network for an unknown name")
    monkeypatch.setattr(species_info, "build", build)
    monkeypatch.setattr(species_info, "find_taxon", boom)
    monkeypatch.setattr(species_info.httpx, "Client", boom)
    return built


def test_unknown_names_fetch_and_store_nothing(client, conn, offline, add_visit):
    add_visit(species="Dark-eyed Junco")
    for name in ("Nothing", "Bushtit", "Dark-eyed%20Junco%20", "dark-eyed%20junco", "%3Cb%3EBushtit"):
        r = client.get(f"/species-info/{name}")
        assert r.status_code == 200 and "No info found" in r.text, name
        assert client.get("/species-photo", params={"name": name}).json() == {"photo": None}
    assert offline == []
    assert conn.execute("SELECT COUNT(*) FROM species_info").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM settings WHERE key LIKE 'refphoto:%'").fetchone()[0] == 0
    # The cached card still shows for any name, with no fetch (a card the mirror was sent, say).
    db.save_species_info(conn, "Bushtit", {"name": "Bushtit", "inat_id": 2, "common": "Bushtit", "links": []}, "2099-01-01T00:00:00")
    assert "No info found" not in client.get("/species-info/Bushtit").text and offline == []


def test_known_names_build_cards(client, conn, offline, add_visit):
    add_visit(species="Dark-eyed Junco")
    add_visit(species="House Finch", corrected_species="Purple Finch")
    add_visit(species="Bushtit", also=json.dumps([{"visitor": "bird", "species": "Black-capped Chickadee", "count": 2}]))
    add_visit(species="Bushtit", corrected_also=json.dumps([{"visitor": "squirrel", "species": "Douglas's Squirrel", "count": 1}]))
    add_visit(visitor="raccoon", species=None)
    known = db.known_species(conn)
    assert known == {"Dark-eyed Junco", "House Finch", "Purple Finch", "Bushtit", "Black-capped Chickadee",
                     "Douglas's Squirrel", "Raccoon"}
    for name in sorted(known):
        assert "No info found" not in client.get(f"/species-info/{name}").text, name
        assert db.get_species_info(conn, name) is not None, name
    assert sorted(offline) == sorted(known)


def test_visitors_never_build_even_known_cards(client, conn, offline, add_visit, locked):
    add_visit(species="Bushtit")
    assert "No info found" in client.get("/species-info/Bushtit").text
    assert offline == [] and db.get_species_info(conn, "Bushtit") is None


def test_species_search_is_gone_on_the_mirror(client, mirror):
    assert client.get("/species-search", params={"q": "finch"}).status_code == 404
