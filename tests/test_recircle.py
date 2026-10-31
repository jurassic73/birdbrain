import io
import json

from PIL import Image

from birdbrain import config, db, identify, recircle
from birdbrain.identify import Circle, Sighting


def test_recircle_adds_circles_only_where_the_answer_agrees_and_never_changes_labels(conn, add_visit, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path)
    buf = io.BytesIO()
    Image.new("RGB", (64, 36)).save(buf, "JPEG")
    (tmp_path / "img.jpg").write_bytes(buf.getvalue())
    day = "2026-10-26"
    junco = add_visit(captured_at=f"{day}T09:00:00")
    squirrel = add_visit(captured_at=f"{day}T10:00:00", visitor="squirrel", species=None, model=identify.SONNET)
    add_visit(captured_at=f"{day}T11:00:00", visitor="none", species=None)  # false trigger: skipped
    add_visit(captured_at="2026-10-25T09:00:00")  # another day: skipped
    asked = []

    def ask(model, images_b64, captured_at, lessons=None, site=None):
        asked.append(model)
        visitor = "bird" if model == identify.HAIKU else "possum"  # the squirrel answer disagrees
        return Sighting(visitor=visitor, species="Bushtit", count=1, confidence="high", notes="",
                        circle=Circle(x=0.5, y=0.5, r=0.1)), {}

    monkeypatch.setattr(identify, "ask", ask)
    assert [v["id"] for v in recircle.pending(conn, day)] == [junco, squirrel]
    for v in recircle.pending(conn, day):
        recircle.recircle(conn, v)
    assert asked == [identify.HAIKU, identify.SONNET]
    v = db.get_visit(conn, junco)
    assert json.loads(v["raw_json"])["calls"]["circle"] == {"x": 0.5, "y": 0.5, "r": 0.1, "claude": True}
    assert v["species"] == "Dark-eyed Junco"
    assert db.get_visit(conn, squirrel)["raw_json"] is None
    assert [v["id"] for v in recircle.pending(conn, day)] == [squirrel]
