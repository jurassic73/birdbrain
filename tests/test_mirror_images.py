"""Every photo the public mirror's pages show must load: behind the friends & family gate, under the /birdbrain
base path, with the photo files in the folders the push script fills. Found by hand once when the mirror showed
broken images; this walks the pages the way a browser would."""
import io
import json
import re

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from helpers.clock import FROZEN_NOW
from birdbrain import config, db
from birdbrain.web import app as webapp
from birdbrain.web import viewer

BASE = "/birdbrain"
# Attributes a browser (or the site's own scripts) load a photo from. Static files use the same ones.
URL_ATTRS = re.compile(r'\s(?:src|href|srcset|data-src|data-full|data-zoom-photo|content)="([^"]+)"')


def jpeg(path, size=(32, 18), color=(90, 120, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    path.write_bytes(buf.getvalue())


@pytest.fixture
def mirror(tmp_path, monkeypatch, conn):
    """The public mirror: BIRDBRAIN_PUBLIC=1, the guest gate on, served under /birdbrain (nginx strips the prefix,
    so the app sees plain paths, and base_path puts the prefix back on the way out)."""
    pw = tmp_path / "viewer.pw"
    pw.write_text(viewer.make_hash("test-password-123") + "\n")
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(pw))
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    monkeypatch.setattr(webapp, "BASE", BASE)
    viewer._fails.clear()
    viewer._cache.clear()
    with TestClient(webapp.app, client=("127.0.0.1", 50000)) as c:
        r = c.post("/guest-login", data={"password": "test-password-123", "next": "/"},
                   headers={"cf-connecting-ip": "203.0.113.77"}, follow_redirects=False)
        assert r.status_code == 303 and viewer.COOKIE in r.headers["set-cookie"]
        # The cookie's path is /birdbrain/ (the browser's view); the app sees the stripped paths, so set it by hand.
        value = r.headers["set-cookie"].split(";")[0].split("=", 1)[1]
        c.cookies.clear()
        c.cookies.set(viewer.COOKIE, value)
        yield c
    viewer._fails.clear()
    viewer._cache.clear()


@pytest.fixture
def photos(conn, add_visit, monkeypatch):
    """Visits whose files sit where the push script puts them (the images folder the app serves as /images, bound
    when the app was imported, so the files go there and are removed again): a plain visit, one whose photo is a
    picked burst frame, one the photo budget cut down to its thumbnail, plus the live frame the Pi pushes."""
    monkeypatch.setattr(webapp, "_live", {"jpeg": None, "at": 0.0, "tried": 0.0})  # no frame cached across tests
    day = config.IMAGES_DIR / "2026" / "11" / "03"
    names = ("a.jpg", "a_t.jpg", "b.jpg", "b_t.jpg", "b_b1.jpg", "b_b2.jpg", "b_b2_t.jpg", "c_t.jpg")
    for name in names:
        jpeg(day / name)
    live = config.DATA_DIR / "live.jpg"
    jpeg(live)
    yield from _photo_visits(conn, add_visit)
    for path in (*(day / n for n in names), live):
        path.unlink(missing_ok=True)


def _photo_visits(conn, add_visit):
    when = FROZEN_NOW.replace(hour=9)
    ids = [add_visit(image_path="2026/11/03/a.jpg", thumb_path="2026/11/03/a_t.jpg",
                     captured_at=when.isoformat(), species="Dark-eyed Junco"),
           add_visit(image_path="2026/11/03/b.jpg", thumb_path="2026/11/03/b_t.jpg", visitor="squirrel",
                     species="Eastern Gray Squirrel", captured_at=when.replace(hour=10).isoformat(),
                     raw_json=json.dumps({"burst": {"best": 0, "fps": 4}})),
           add_visit(image_path="2026/11/03/c.jpg", thumb_path="2026/11/03/c_t.jpg", species="Bushtit",
                     captured_at=when.replace(hour=11).isoformat())]
    db.set_photo_frame(conn, ids[1], 2)
    conn.execute("UPDATE visits SET photo_level = 'thumb' WHERE id = ?", (ids[2],))
    db.set_value(conn, db.SPECIES_PHOTO, json.dumps({"Dark-eyed Junco": ids[0]}))
    conn.commit()
    yield ids


def photo_urls(html: str) -> set[str]:
    """Every photo URL a page asks for: visit photos, thumbnails, burst frames and the live frame."""
    found = set()
    for value in URL_ATTRS.findall(html):
        for part in value.split(","):  # srcset: "url 1x, url 2x"
            url = part.strip().split(" ")[0]
            if "/images/" in url or "/live/" in url or "/live.jpg" in url or url.endswith(".jpg"):
                found.add(url)
    return found


def test_every_photo_on_the_public_pages_loads_under_the_base_path(mirror, photos):
    v1, v2, v3 = photos
    pages = ["/", "/live", f"/visit/{v1}", f"/visit/{v2}", f"/visit/{v3}", "/species", "/species/Dark-eyed%20Junco",
             "/species/Eastern%20Gray%20Squirrel", "/stats", "/about"]
    urls: dict[str, set[str]] = {}
    for page in pages:
        r = mirror.get(page)
        assert r.status_code == 200, page
        urls[page] = photo_urls(r.text)
    assert urls["/"] and urls[f"/visit/{v1}"] and urls["/species"] and urls["/species/Dark-eyed%20Junco"]
    # The picked burst frame and its thumbnail, and the thumbnail-only visit, are what those pages show.
    assert any(u.endswith("/2026/11/03/b_b2.jpg") for u in urls[f"/visit/{v2}"])
    assert any(u.endswith("/2026/11/03/c_t.jpg") for u in urls[f"/visit/{v3}"])
    assert any(u.endswith("/2026/11/03/b_b2_t.jpg") for u in urls["/"])
    failures = []
    for page, found in urls.items():
        for url in sorted(found):
            if not url.startswith(BASE + "/"):
                failures.append(f"{page}: {url} lacks the base path")
                continue
            r = mirror.get(url[len(BASE):])  # nginx strips the prefix before the app sees it
            if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
                failures.append(f"{page}: {url} -> {r.status_code} {r.headers.get('content-type')}")
            elif "private" not in r.headers.get("cache-control", "") and "no-store" not in r.headers.get("cache-control", ""):
                failures.append(f"{page}: {url} may be kept by a shared cache: {r.headers.get('cache-control')!r}")
    assert not failures, "\n".join(failures)


def test_gate_redirects_and_photos_are_never_kept_by_a_shared_cache(mirror, photos):
    """Cloudflare caches .jpg URLs by default whatever cookie the request carried. A cookieless fetch of a photo
    URL gets the sign-in redirect; if a CDN kept that, every guest would get it as the photo for minutes."""
    photo = "/images/2026/11/03/a.jpg"
    signed = mirror.get(photo)
    assert signed.status_code == 200 and signed.headers["cache-control"] == "private"
    assert mirror.get("/").headers["cache-control"] == "private"
    assert mirror.get("/live/latest.jpg").headers["cache-control"] == "no-store"  # as before, never stale either
    # Static files are open and versioned (?v=): they keep their normal caching.
    assert "private" not in mirror.get("/static/style.css").headers.get("cache-control", "")
    cookie = mirror.cookies.get(viewer.COOKIE)
    mirror.cookies.clear()
    anon = mirror.get(photo, follow_redirects=False)
    assert anon.status_code == 303 and anon.headers["location"].startswith(BASE + "/guest-login")
    assert anon.headers["cache-control"] == "no-store"
    assert mirror.post("/visit/1/delete", follow_redirects=False).headers["cache-control"] == "no-store"
    mirror.cookies.set(viewer.COOKIE, cookie)
    assert mirror.get(photo).status_code == 200


def test_the_mirror_never_makes_the_photo_folder_itself(tmp_path, monkeypatch):
    """On the mirror the push user owns images/ and the pushes give it the service's group through the setgid
    bit. A folder the service made at startup would be the service's own (no write for the push user) and stop
    every push, so only the Pi's site makes it; the mirror serves 404s until the first push fills it."""
    folder = tmp_path / "data" / "images"
    monkeypatch.setattr(config, "IMAGES_DIR", folder)
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    assert webapp.images_dir() == folder and not folder.exists()
    # The mount the app uses serves a missing folder as 404s rather than erroring, and picks it up once the
    # first push has made it, without a restart.
    from fastapi import FastAPI

    site = FastAPI()
    site.mount("/images", webapp.PhotoFiles(directory=folder, check_dir=False), name="images")
    c = TestClient(site)
    assert c.get("/images/2026/11/03/a.jpg").status_code == 404
    jpeg(folder / "2026" / "11" / "03" / "a.jpg")
    assert c.get("/images/2026/11/03/a.jpg").status_code == 200
    monkeypatch.delenv("BIRDBRAIN_PUBLIC")
    assert webapp.images_dir() == folder and folder.is_dir()


def _json_site_urls(data) -> list[str]:
    """Every site URL (an absolute path such as "/images/...") inside a JSON answer, at any depth."""
    found = []
    if isinstance(data, dict):
        for v in data.values():
            found += _json_site_urls(v)
    elif isinstance(data, list):
        for v in data:
            found += _json_site_urls(v)
    elif isinstance(data, str) and (data.startswith("/") or "/images/" in data or "/static/" in data):
        found.append(data)
    return found


def test_json_answers_carry_the_base_path_and_resolve(mirror, photos, conn, add_visit):
    """The base_path middleware rewrites HTML only. Routes that answer a URL in JSON (for a script to put in an
    img src) must add the prefix themselves (app.site_url), or the browser asks the host's root and gets nothing."""
    new = add_visit(image_path="2026/11/03/a.jpg", thumb_path="2026/11/03/a_t.jpg", species="Dark-eyed Junco",
                    captured_at=FROZEN_NOW.replace(hour=11, minute=30).isoformat())
    conn.commit()
    compare_url = f"/species/Dark-eyed%20Junco/cover-compare?visit_id={new}"
    assert mirror.get(compare_url).status_code == 404  # the star's question is the admin's: not on the mirror
    answers = {"/manifest.webmanifest": mirror.get("/manifest.webmanifest")}
    manifest = answers["/manifest.webmanifest"].json()
    assert manifest["start_url"] == BASE + "/" and manifest["scope"] == BASE + "/"
    failures = []
    for route, r in answers.items():
        assert r.status_code == 200, route
        urls = _json_site_urls(r.json())
        assert urls, route
        for url in urls:
            if not url.startswith(BASE + "/"):
                failures.append(f"{route}: {url} lacks the base path")
                continue
            got = mirror.get(url[len(BASE):])  # nginx strips the prefix before the app sees it
            want = "text/html" if url.endswith("/") else "image/"  # start_url is a page, the rest are pictures
            if got.status_code != 200 or not got.headers.get("content-type", "").startswith(want):
                failures.append(f"{route}: {url} -> {got.status_code} {got.headers.get('content-type')}")
    assert not failures, "\n".join(failures)


def test_admin_json_carries_the_base_path_too(mirror, photos, conn, add_visit, monkeypatch):
    """The admin's JSON with a photo URL in it (the star's "Replace the cover?" question) under a base path: the same
    site, not public, no password, so everyone is admin. The mirror itself answers it 404 (test above)."""
    v1, _, _ = photos
    new = add_visit(image_path="2026/11/03/a.jpg", thumb_path="2026/11/03/a_t.jpg", species="Dark-eyed Junco",
                    captured_at=FROZEN_NOW.replace(hour=11, minute=30).isoformat())
    conn.commit()
    monkeypatch.delenv("BIRDBRAIN_PUBLIC")
    compare = mirror.get(f"/species/Dark-eyed%20Junco/cover-compare?visit_id={new}")
    assert compare.status_code == 200
    assert compare.json()["cover"]["id"] == v1 and compare.json()["candidate"]["id"] == new
    urls = _json_site_urls(compare.json())
    assert urls
    for url in urls:
        assert url.startswith(BASE + "/"), url
        got = mirror.get(url[len(BASE):])
        assert got.status_code == 200 and got.headers["content-type"].startswith("image/"), url


def test_json_answers_keep_plain_paths_without_a_base_path(conn, add_visit):
    """On the LAN (no BIRDBRAIN_BASE) the URLs stay as they were."""
    assert webapp.BASE == ""
    client = TestClient(webapp.app)
    old = add_visit(species="Bushtit", image_path="2026/11/03/old.jpg")
    new = add_visit(species="Bushtit", image_path="2026/11/03/new.jpg")
    db.set_species_photo(conn, "Bushtit", old)
    r = client.get(f"/species/Bushtit/cover-compare?visit_id={new}").json()
    assert r["cover"]["src"] == "/images/2026/11/03/old.jpg"
    assert r["candidate"]["src"] == "/images/2026/11/03/new.jpg"
    m = client.get("/manifest.webmanifest").json()
    assert m["start_url"] == "/" and all(i["src"].startswith("/static/") for i in m["icons"])
    assert webapp.site_url("//evil.example/x") == "//evil.example/x"  # protocol-relative: never prefixed
