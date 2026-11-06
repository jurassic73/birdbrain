"""The owner's rule: "There should be zero admin functionality exposed on mirror."

On the public mirror (BIRDBRAIN_PUBLIC=1, behind the friends & family sign-in) every route that isn't one of its
public pages answers the same 404 as a path that doesn't exist: the admin pages, the login, the admin's JSON, Claude
and camera calls, and every change (only the guest sign-in takes a POST). The pages it does serve carry no admin
control, link or script, hidden or not.

The app denies by default (auth.MIRROR_ROUTES). This file keeps its own list of both kinds, so a new route fails
test_every_route_is_classified until someone decides whether the mirror may serve it."""

import io
import json
import re

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from helpers.clock import FROZEN_NOW
from helpers.public_no_admin import ADMIN, PUBLIC
from birdbrain import config, db
from birdbrain.web import app as webapp, auth, viewer


DAY = "2026/11/02"  # the photo folder these tests write into (and clean up)


# Admin controls and the admin's pages and endpoints, as they appear in the HTML. None may be on a mirror page,
# not even hidden or inside a script. (Prose that names an admin feature, such as About's "pressed Make public" or a
# tip's "(Use as photo)", is public information; the controls themselves are what's checked.)
MARKERS = (
    "hx-post", "hx-put", "hx-delete", 'method="post"', ">Admin<", "admin-btn", "Log out", "data-admin",
    "/settings", "/review", "/false-triggers", "/notes", "/lessons", "/login", "/logout", "/calibration/",
    "/correct", "/ask-species", "/check-circle", "/clear-correction", "/reviewed", "/unconfirm", "/keep\"",
    "/public\"", "/private\"", "/delete", "/photo-frame", "/sound-broken\"", "species-search", "species-photo",
    "photo-reminder", "cover-compare", "/species/Bushtit/photo",
    'class="review"', "form.review", "review-fold", "Correct it", "correct-open", "make-public",
    "use-frame", "use-claude", "draw-note", "Add circles", "fav-star", "confirm-btn",
    "Confirm no animal", "Delete visit", "check-btn", "data-source=", "lessons-popup",
    "star.js", "scene-edit.js", "save-feedback.js", "cam-live.js", "embed-close",
)


def jpeg(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("RGB", (32, 18), (90, 120, 60)).save(buf, "JPEG")
    path.write_bytes(buf.getvalue())


@pytest.fixture
def seeded(conn, add_visit, monkeypatch):
    """Visits the mirror would get: a settled one with its owner's circle check and a burst (a false trigger the owner
    corrected to an animal keeps its burst), one sent to review, a species cover, and their photo files, plus the
    live frame the Pi pushes. The files go into the folder the app serves as /images and are removed again."""
    monkeypatch.setattr(webapp, "_live", {"jpeg": None, "at": 0.0, "tried": 0.0})
    day = config.IMAGES_DIR / DAY
    names = ("a.jpg", "a_t.jpg", "a_b1.jpg", "a_b2.jpg", "b.jpg", "b_t.jpg")
    for name in names:
        jpeg(day / name)
    live = config.DATA_DIR / "live.jpg"
    jpeg(live)
    raw = {"calls": {"haiku": {"frames": 1, "answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}},
                     "circle": {"x": 0.5, "y": 0.5, "r": 0.1},
                     "owner_checks": [{"source": "haiku", "say": "Bushtit", "answer": {"notes": "small grey bird"},
                                       "at": FROZEN_NOW.replace(hour=9, minute=5).isoformat(), "usd": 0.001}]},
           "trigger": "rim", "burst": {"best": 0, "fps": 4}}
    settled = add_visit(image_path=f"{DAY}/a.jpg", thumb_path=f"{DAY}/a_t.jpg", visitor="none", species=None,
                        corrected_visitor="bird", corrected_species="Bushtit", confirmed_at=FROZEN_NOW.isoformat(),
                        captured_at=FROZEN_NOW.replace(hour=9).isoformat(), raw_json=json.dumps(raw),
                        marks=json.dumps([{"x": 0.4, "y": 0.4, "r": 0.1, "label": "Bushtit"}]))
    review = add_visit(image_path=f"{DAY}/b.jpg", thumb_path=f"{DAY}/b_t.jpg", species="Bushtit", confidence="low",
                       needs_review=1, captured_at=FROZEN_NOW.replace(hour=10).isoformat(), raw_json=json.dumps(raw))
    db.set_star(conn, "Bushtit", settled, "cover")
    conn.commit()
    yield {"visit_id": settled, "review_id": review}
    for path in (*(day / n for n in names), live):
        path.unlink(missing_ok=True)


@pytest.fixture
def mirror(tmp_path, monkeypatch, conn, seeded):
    """The public mirror as it runs: BIRDBRAIN_PUBLIC=1 (with an admin password set, which must change nothing), the
    friends & family gate on and a guest signed in."""
    pw = tmp_path / "viewer.pw"
    pw.write_text(viewer.make_hash("test-password-123") + "\n")
    monkeypatch.setenv("VIEWER_PASSWORD_FILE", str(pw))
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret!")

    async def no_camera():
        raise AssertionError("the public mirror contacted the camera")
    monkeypatch.setattr(webapp, "grab_live_frame", no_camera)
    viewer._fails.clear()
    viewer._cache.clear()
    with TestClient(webapp.app, client=("127.0.0.1", 50000)) as c:
        r = c.post("/guest-login", data={"password": "test-password-123", "next": "/"}, follow_redirects=False)
        assert r.status_code == 303 and viewer.COOKIE in r.headers["set-cookie"]
        yield c
    viewer._fails.clear()
    viewer._cache.clear()


def fill(template: str, seeded: dict) -> str:
    """A concrete URL for a route template, with a real visit id and species name."""
    if template == "/static":
        return "/static/style.css"
    if template == "/images":
        return f"/images/{DAY}/a_b1.jpg"
    values = {"visit_id": str(seeded["visit_id"]), "name": "Bushtit", "source": "haiku", "state": "off",
              "change_id": "1", "cal_id": "1"}
    url = re.sub(r"\{(\w+)\}", lambda m: values[m.group(1)], template)  # a KeyError: give the new parameter a value
    return url


def test_every_admin_route_and_every_change_is_a_404_on_the_mirror(mirror, seeded, conn):
    nowhere = mirror.get("/no-such-page")
    assert nowhere.status_code == 404
    checked = []
    for method, template in sorted(ADMIN | {(m, p) for _, p in PUBLIC for m in ("POST", "PUT", "PATCH", "DELETE")}):
        if (method, template) in PUBLIC:
            continue  # the guest sign-in
        assert not auth.on_mirror(method, template), (method, template)
        url = fill(template, seeded)
        r = mirror.request(method, url, follow_redirects=False)
        # The very answer a missing page gets: not a login, a 403 or a read-only page.
        assert (r.status_code, r.content) == (404, nowhere.content), (method, url, r.status_code)
        checked.append((method, template))
    assert len(checked) > len(ADMIN)
    assert db.get_visit(conn, seeded["visit_id"])["corrected_species"] == "Bushtit"  # nothing changed


def test_admin_paths_look_like_missing_pages_to_someone_not_signed_in(mirror):
    """Before the guest sign-in every path, there or not, gets the same redirect to it: nothing tells them apart."""
    mirror.cookies.clear()
    missing = mirror.get("/no-such-page", follow_redirects=False)
    assert missing.status_code == 303
    for path in ("/settings", "/login", "/notes", "/false-triggers", "/species-search"):
        r = mirror.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == f"/guest-login?next={path.replace('/', '%2F')}"


def test_the_public_routes_answer_on_the_mirror(mirror, seeded):
    for method, template in sorted(PUBLIC):
        if method != "GET":
            continue
        url = fill(template, seeded)
        r = mirror.get(url, follow_redirects=False)
        assert r.status_code < 400, (url, r.status_code)


def pages(seeded: dict) -> list[str]:
    vid, rid = seeded["visit_id"], seeded["review_id"]
    return ["/", "/live", "/scoreboard", "/visits/more", f"/visit/{vid}", f"/visit/{rid}", f"/visit/{vid}?embed=1",
            f"/visit/{vid}?ft=1", "/species", "/species/Bushtit", "/species/Bushtit/more", "/species-info/Bushtit",
            "/species-info/Bushtit?mini=1", "/species-info/Bushtit?fold=1", "/stats", "/weather", "/about",
            "/guest-login"]


def test_mirror_pages_carry_no_admin_controls_links_or_scripts(mirror, seeded):
    found = []
    for url in pages(seeded):
        r = mirror.get(url)
        assert r.status_code == 200, url
        found += [f"{url}: {m}" for m in MARKERS if m in r.text]
    assert not found, "\n".join(found)


def test_mirror_keeps_the_public_information(mirror, seeded):
    """Read-only information the owner chose to show stays: how Claude decided, the owner's circle checks (Settings >
    Public mirror), the burst filmstrip of a visit that kept one, the tour."""
    page = mirror.get(f"/visit/{seeded['visit_id']}").text
    assert "How it was decided" in page and "Your circle checks" in page
    assert f"{DAY}/a_b1.jpg" in page and f"{DAY}/a_b2.jpg" in page
    assert "Run the tour again" in mirror.get("/about").text


def test_the_admin_still_has_everything_on_the_pi(seeded, conn):
    """The same pages on the Pi with no password (everyone is admin): the markers above are really the admin's."""
    with TestClient(webapp.app) as c:
        page = c.get(f"/visit/{seeded['review_id']}").text
        for marker in ("hx-post", "/correct", "check-btn", "fav-star", "star.js", "Delete visit", "/settings"):
            assert marker in page, marker
        assert 'href="/settings#zones"' in c.get("/about").text
