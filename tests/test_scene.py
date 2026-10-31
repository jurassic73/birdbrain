import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from birdbrain import scene
from birdbrain.scene import Ellipse, Outline, Point, SceneObject
from birdbrain.web import app as webapp


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1920, 1080), (90, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


def bath_outline() -> Outline:
    return Outline(notes="Glare on the right in the morning.", objects=[
        SceneObject(name="birdbath", kind="birdbath", shape="ellipse", ellipse=Ellipse(cx=0.5, cy=0.5, rx=0.2, ry=0.1),
                    points=[], note=""),
        SceneObject(name="lawn", kind="ground", shape="polygon", ellipse=None,
                    points=[Point(x=-0.1, y=0.6), Point(x=1.2, y=0.6), Point(x=0.5, y=1.0)], note="grass"),
        SceneObject(name="nothing", kind="perch", shape="polygon", ellipse=None, points=[Point(x=0, y=0)], note=""),
    ])


def fake_ask(monkeypatch, result=None, seen=None):
    def ask(jpeg_, brief, location=""):
        if seen is not None:
            seen.append((brief, location))
        return result, {"model": scene.MODEL, "input_tokens": 1800, "output_tokens": 400}
    monkeypatch.setattr(scene, "ask", ask)


def test_clean_clamps_to_the_frame_and_drops_shapeless_objects():
    objs = scene.clean(bath_outline())
    assert [o["name"] for o in objs] == ["birdbath", "lawn"]
    assert objs[0]["ellipse"] == [0.5, 0.5, 0.2, 0.1]
    assert objs[1]["points"] == [[0.0, 0.6], [1.0, 0.6], [0.5, 1.0]]


def test_outline_saves_a_draft_until_accepted(conn, monkeypatch):
    seen = []
    fake_ask(monkeypatch, bath_outline(), seen)
    d = scene.outline(conn, jpeg(), "  A birdbath on a lawn ", "Vancouver, WA")
    assert seen == [("  A birdbath on a lawn ", "Vancouver, WA")]
    assert scene.draft(conn)["brief"] == "A birdbath on a lawn"
    assert scene.draft(conn)["objects"] == d["objects"] and scene.accepted(conn) is None
    assert d["usd"] == round((1800 * 2.0 + 400 * 10.0) / 1e6, 4)
    assert (scene.picture_dir() / d["image"]).exists()

    assert scene.accept(conn)
    assert scene.draft(conn) is None and scene.accepted(conn)["objects"] == d["objects"]
    assert (scene.picture_dir() / d["image"]).exists()  # the accepted scene keeps its frame


def test_a_refusal_leaves_an_empty_draft_that_cant_be_accepted(conn, monkeypatch):
    fake_ask(monkeypatch, None)
    d = scene.outline(conn, jpeg(), "a feeder")
    assert d["objects"] == [] and "Try again" in d["notes"]
    assert not scene.accept(conn)


def test_discarding_removes_the_drafts_frame(conn, monkeypatch):
    fake_ask(monkeypatch, bath_outline())
    d = scene.outline(conn, jpeg(), "a birdbath")
    scene.discard(conn)
    assert scene.draft(conn) is None and not (scene.picture_dir() / d["image"]).exists()


def test_settings_page_outlines_and_accepts_a_scene(client, conn, monkeypatch):
    fake_ask(monkeypatch, bath_outline())

    async def grab():
        return jpeg()
    monkeypatch.setattr(webapp, "grab_main_frame", grab)

    assert "Save and outline" in client.get("/notes").text
    r = client.post("/notes/scene/outline", data={"scene_notes": "A birdbath"}, headers={"HX-Request": "true"})
    assert r.status_code == 200 and "does this look right?" in r.text and "Glare on the right" in r.text
    image = scene.draft(conn)["image"]
    assert client.get(f"/calibration/{image}").status_code == 200

    r = client.post("/notes/scene/accept", headers={"HX-Request": "true"})
    assert "Your scene" in r.text and "does this look right?" not in r.text
    page = client.get("/notes").text
    assert "2 objects outlined" in page and "Save and outline again" in page


def test_outline_reports_a_camera_that_is_down(client, conn, monkeypatch):
    async def grab():
        return b""
    monkeypatch.setattr(webapp, "grab_main_frame", grab)
    monkeypatch.setitem(webapp._live, "jpeg", None)
    monkeypatch.setattr(webapp, "live_file", lambda: scene.picture_dir() / "missing.jpg")
    r = client.post("/notes/scene/outline", data={"scene_notes": "x"}, headers={"HX-Request": "true"})
    assert "get a frame from the camera" in r.text


def test_scene_pictures_only_by_name(client):
    assert client.get("/calibration/scene_..%2Fsecret.jpg").status_code == 404


def test_the_owner_can_drag_a_shape_and_reset_it(client, conn, monkeypatch):
    fake_ask(monkeypatch, bath_outline())
    scene.outline(conn, jpeg(), "a birdbath")
    bath = scene.draft(conn)["objects"][0]
    r = client.post("/notes/scene/shape/draft/0", json={"ellipse": [0.45, 0.5, 0.18, 0.12]})
    assert r.status_code == 200 and r.json()["ellipse"] == [0.45, 0.5, 0.18, 0.12]
    edited = scene.draft(conn)["objects"][0]
    assert edited["edited"] and edited["auto"]["ellipse"] == bath["ellipse"]
    page = client.get("/notes/scene/panel").text
    assert "adjusted by you" in page and 'hx-post="/notes/scene/reset/draft/0"' in page
    assert "hx-confirm" in client.get("/notes").text  # outlining again would drop the adjustment

    # an outline needs 3 corners; a shape is an ellipse or points, not both; water goes with a rim
    assert client.post("/notes/scene/shape/draft/1", json={"points": [[0, 0], [1, 1]]}).status_code == 400
    assert client.post("/notes/scene/shape/draft/0", json={"ellipse": [0.5, 0.5, 0.1, 0.1],
                                                            "points": [[0, 0], [1, 0], [1, 1]]}).status_code == 400
    assert client.post("/notes/scene/shape/draft/0", json={"inner": [0.5, 0.5, 0.1, 0.1]}).status_code == 400
    assert client.post("/notes/scene/shape/draft/9", json={"ellipse": [0.5, 0.5, 0.1, 0.1]}).status_code == 404
    assert client.post("/notes/scene/shape/nope/0", json={"ellipse": [0.5, 0.5, 0.1, 0.1]}).status_code == 404
    clamped = client.post("/notes/scene/shape/draft/0", json={"ellipse": [2, -1, 0.1, 0.1]}).json()
    assert clamped["ellipse"] == [1.0, 0.0, 0.1, 0.1]

    client.post("/notes/scene/reset/draft/0", headers={"HX-Request": "true"})
    back = scene.draft(conn)["objects"][0]
    assert back["ellipse"] == bath["ellipse"] and "edited" not in back and "auto" not in back


def test_the_accepted_scene_can_be_adjusted_too(client, conn, monkeypatch):
    fake_ask(monkeypatch, bath_outline())
    scene.outline(conn, jpeg(), "a birdbath")
    scene.accept(conn)
    lawn = scene.accepted(conn)["objects"][1]
    moved = [[x, round(min(y + 0.05, 1), 4)] for x, y in lawn["points"]]
    assert client.post("/notes/scene/shape/scene/1", json={"points": moved}).status_code == 200
    assert scene.accepted(conn)["objects"][1]["points"] == moved


def test_a_shape_switches_between_ellipse_and_points_and_reset_restores_it(client, conn, monkeypatch):
    fake_ask(monkeypatch, bath_outline())
    scene.outline(conn, jpeg(), "a birdbath")
    bath = scene.draft(conn)["objects"][0]
    r = client.post("/notes/scene/convert/draft/0/points")
    assert r.status_code == 200 and len(r.json()["points"]) == scene.CORNERS and "ellipse" not in r.json()
    # the corners can now be dragged one by one
    pts = r.json()["points"]
    pts[0] = [0.8, 0.5]
    assert client.post("/notes/scene/shape/draft/0", json={"points": pts}).status_code == 200
    back = client.post("/notes/scene/convert/draft/0/ellipse").json()
    assert "points" not in back and back["ellipse"][2] > 0
    assert client.post("/notes/scene/convert/draft/0/blob").status_code == 400
    client.post("/notes/scene/reset/draft/0", headers={"HX-Request": "true"})
    obj = scene.draft(conn)["objects"][0]
    assert obj["ellipse"] == bath["ellipse"] and "points" not in obj and "edited" not in obj


def test_a_shape_switched_to_points_on_the_page_saves_in_one_call(client, conn, monkeypatch):
    fake_ask(monkeypatch, bath_outline())
    scene.outline(conn, jpeg(), "a birdbath")
    pts = [[0.4, 0.45], [0.6, 0.45], [0.65, 0.55], [0.5, 0.6], [0.35, 0.55]]
    r = client.post("/notes/scene/shape/draft/0", json={"points": pts})
    assert r.status_code == 200 and r.json() == {"points": pts}
    obj = scene.draft(conn)["objects"][0]
    assert "ellipse" not in obj and obj["auto"]["ellipse"]  # Reset can still bring the ellipse back
