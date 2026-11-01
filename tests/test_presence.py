import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from birdbrain import classifier, db, presence, watch
from birdbrain.web import app as webapp

DIM = 16


def vec(animal: bool, rng) -> np.ndarray:
    """Fake embeddings: animals lean one way, empty frames the other."""
    v = rng.normal(0, 0.3, DIM)
    v[0] += 1.0 if animal else -1.0
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def fresh_model(monkeypatch):
    monkeypatch.setattr(presence, "_model", None)


@pytest.fixture
def yard(conn, add_visit):
    """12 visitors and 40 empty frames with embeddings."""
    rng = np.random.default_rng(1)
    for i in range(52):
        animal = i < 12
        vid = add_visit(**({} if animal else {"visitor": "none", "species": None}))
        presence.store(conn, vid, vec(animal, rng))
    return rng


def test_too_few_labels_means_no_verdict(conn, add_visit):
    presence.store(conn, add_visit(), np.ones(DIM))
    assert presence.check(conn, np.ones(DIM)) == {"model": presence.MODEL_NAME, "score": None, "empty": False,
                                                 "note": "not enough labelled frames yet"}


def test_it_learns_empty_from_animal_and_keeps_every_visitor(conn, yard):
    m = presence.model(conn)
    assert m["visitors"] == 12 and m["empty"] == 40 and 0 < m["threshold"] <= presence.MAX_THRESHOLD
    assert presence.check(conn, vec(False, yard))["empty"] is True
    assert presence.check(conn, vec(True, yard))["empty"] is False


def test_it_retrains_when_a_label_changes(conn, yard, add_visit):
    first = presence.model(conn)
    assert presence.model(conn) is first  # cached while nothing changed
    empty = conn.execute("SELECT id FROM visits WHERE visitor = 'none' LIMIT 1").fetchone()[0]
    db.correct_visit(conn, empty, "squirrel", None, None)  # the owner found an animal in a false trigger
    second = presence.model(conn)
    assert second is not first and second["visitors"] == 13
    db.confirm_visits(conn, [empty + 1])
    assert presence.model(conn) is not second  # a confirm retrains too (it counts extra)


def test_gate_skips_claude_and_keeps_the_frame_for_review(monkeypatch, conn, yard):
    burst = [cv2.imencode(".jpg", np.full((1080, 1920, 3), 90, np.uint8))[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "gate")
    emb = vec(False, yard)
    monkeypatch.setattr(classifier, "box_embedding", lambda jpeg, box: emb)
    asked = []
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: asked.append(1))
    assert watch.handle_event() == "none"
    assert asked == []
    v = db.latest_visit(conn, real_only=False)
    raw = json.loads(v["raw_json"])
    assert v["model"] == presence.MODEL_NAME and raw["calls"]["skipped_claude"] and raw["calls"]["presence"]["empty"]
    stored = conn.execute("SELECT vec FROM embeddings WHERE visit_id = ?", (v["id"],)).fetchone()
    assert stored is not None  # it becomes training data once checked


def test_shadow_records_the_score_and_still_asks_claude(monkeypatch, conn, yard):
    from birdbrain.identify import Result, Sighting
    burst = [cv2.imencode(".jpg", np.full((1080, 1920, 3), 90, np.uint8))[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    monkeypatch.setattr(classifier, "box_embedding", lambda jpeg, box: vec(False, yard))
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: Result(
        Sighting(visitor="none", species=None, count=0, confidence="high", notes=""), "claude-haiku-5-5", False, False, {}))
    assert watch.handle_event() == "none"
    raw = json.loads(db.latest_visit(conn, real_only=False)["raw_json"])
    assert raw["calls"]["presence"]["empty"] and "skipped_claude" not in raw["calls"]
    r = db.presence_report(conn)
    assert (r["frames"], r["empty"], r["would_skip"], r["skipped"]) == (1, 1, 1, 0)


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


def test_looks_right_confirms_and_undo_takes_it_back(client, conn, add_visit):
    vid = add_visit()
    assert "Looks right" in client.get(f"/visit/{vid}").text
    client.post(f"/visit/{vid}/reviewed")
    assert db.get_visit(conn, vid)["confirmed_at"]
    assert "Checked:" in client.get(f"/visit/{vid}").text
    client.post(f"/visit/{vid}/unconfirm")
    assert db.get_visit(conn, vid)["confirmed_at"] is None


def test_false_triggers_confirm_all_shown_skips_real_visits(client, conn, add_visit):
    empty = [add_visit(visitor="none", species=None) for _ in range(3)]
    real = add_visit()
    page = client.get("/false-triggers").text
    assert "3 not checked yet" in page and ">Confirm</button>" in page
    assert "no animal was found: the local animal check on the Pi called it empty" in page
    r = client.post("/false-triggers/confirm", data={"ids": ",".join(map(str, [*empty[:2], real]))},
                    follow_redirects=False)
    assert r.headers["location"] == "/false-triggers?confirmed=2"
    assert [bool(db.get_visit(conn, i)["confirmed_at"]) for i in [*empty, real]] == [True, True, False, False]
    assert db.count_unchecked_false_triggers(conn) == 1


def test_settings_switches_the_animal_check(client, conn):
    assert "Local checks" in client.get("/settings").text
    client.post("/settings/presence", data={"mode": "gate"})
    assert presence.mode(conn) == "gate"
    assert client.post("/settings/presence", data={"mode": "nope"}).status_code == 400
