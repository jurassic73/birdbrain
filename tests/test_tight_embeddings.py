"""The tight views' BioCLIP 2 embeddings (fingerprints) kept per visit as training data: nothing reads them yet."""
import json
import sqlite3

import cv2
import numpy as np
import pytest

from birdbrain import classifier, config, db, presence, watch
from helpers.classifier import (LABELS, LOCAL, MOTION_CIRCLE, FakeClassifier, gate_yard, one_hot,  # noqa: F401
                                sure)

real_store = presence.store
DIM = 8


def vec(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def stored(conn, vid):
    return conn.execute("SELECT * FROM embeddings WHERE visit_id = ?", (vid,)).fetchone()


def decode(blob):
    return np.frombuffer(blob, np.float16).astype(np.float32)


def last_id(conn):
    return conn.execute("SELECT MAX(id) FROM visits").fetchone()[0]


# --- schema and the helper ------------------------------------------------------------------------------------

def test_the_migration_adds_the_columns_to_an_existing_database(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    conn.execute("DROP TABLE embeddings")  # the table as it was before the tight columns
    conn.execute("""CREATE TABLE embeddings (visit_id INTEGER PRIMARY KEY REFERENCES visits(id) ON DELETE CASCADE,
                    vec BLOB NOT NULL, view INTEGER, animal_vec BLOB, animal_frame INTEGER, animal_score REAL,
                    animal_by_owner INTEGER)""")
    conn.execute("INSERT INTO visits (captured_at, image_path, thumb_path, visitor, model) "
                 "VALUES ('2026-11-01T10:00:00', 'a', 'b', 'bird', 'm')")
    conn.execute("INSERT INTO embeddings (visit_id, vec, view) VALUES (1, ?, 2)", (b"abcd",))
    conn.commit()
    assert "tight_vec" not in {r["name"] for r in conn.execute("PRAGMA table_info(embeddings)")}
    db.init_db(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(embeddings)")}
    assert {"tight_vec", "tight_view", "circle_vec", "circle_view", "circle_model"} <= cols
    row = conn.execute("SELECT * FROM embeddings").fetchone()
    assert row["vec"] == b"abcd" and row["view"] == 2 and row["tight_vec"] is None
    db.init_db(conn)  # again: nothing to add, no error


def test_store_tight_fills_only_its_columns(conn, add_visit):
    vid = add_visit()
    real_store(conn, vid, vec(0), 2)
    conn.execute("UPDATE embeddings SET animal_vec = ?, animal_frame = 3, animal_score = 0.9, animal_by_owner = 1",
                 (b"xx",))
    before = dict(stored(conn, vid))
    assert presence.store_tight(conn, vid, tight=vec(1)) == 1
    row = stored(conn, vid)
    assert np.allclose(decode(row["tight_vec"]), vec(1)) and row["tight_view"] == classifier.TIGHT_VIEW
    assert row["circle_vec"] is None and row["circle_model"] is None
    assert presence.store_tight(conn, vid, circle=vec(2), circle_model="sonnet") == 1
    row = stored(conn, vid)
    assert np.allclose(decode(row["tight_vec"]), vec(1))  # kept
    assert np.allclose(decode(row["circle_vec"]), vec(2)) and row["circle_model"] == "sonnet"
    for col in ("vec", "view", "animal_vec", "animal_frame", "animal_score", "animal_by_owner"):
        assert row[col] == before[col]
    assert len(row["tight_vec"]) == DIM * 2  # float16


def test_store_tight_never_makes_a_row_and_skips_bad_vectors(conn, add_visit):
    vid = add_visit()
    assert presence.store_tight(conn, vid, tight=vec(1)) == 0  # no wide embedding: nothing to attach to
    assert stored(conn, vid) is None
    real_store(conn, vid, vec(0))
    bad = vec(1)
    bad[2] = float("nan")
    assert presence.store_tight(conn, vid, tight=bad, circle=np.array([])) == 0
    assert stored(conn, vid)["tight_vec"] is None


def test_the_new_columns_do_not_change_the_pi_animal_check_training(conn, add_visit):
    ids = [add_visit(), add_visit(visitor="none", species=None)]
    for i, vid in enumerate(ids):
        real_store(conn, vid, vec(i), classifier.PRESENCE_VIEW)
    before = presence.examples(conn)
    for vid in ids:
        presence.store_tight(conn, vid, tight=vec(5), circle=vec(6), circle_model="haiku")
    after = presence.examples(conn)
    assert np.array_equal(before["ids"], after["ids"]) and np.array_equal(before["X"], after["X"])
    assert np.array_equal(before["y"], after["y"])


def test_deleting_a_visit_deletes_its_fingerprints(conn, add_visit):
    vid = add_visit()
    real_store(conn, vid, vec(0))
    presence.store_tight(conn, vid, tight=vec(1), circle=vec(2), circle_model="haiku")
    db.delete_visit(conn, vid)
    assert stored(conn, vid) is None


def test_the_mirror_empties_the_whole_embeddings_table():
    from birdbrain import mirror
    assert "embeddings" in mirror.EMPTIED_TABLES


# --- classify and with_embedding ------------------------------------------------------------------------------

class LookClassifier(FakeClassifier):
    def look(self, bgr):
        if self.fail:
            raise RuntimeError("model broke")
        self.shapes.append(bgr.shape)
        return vec(3), self._probs


def test_with_embedding_returns_the_tight_squares_embedding_and_leaves_the_guess_alone():
    img = np.full((1080, 1920, 3), 100, np.uint8)
    clf = LookClassifier()
    circle = {"x": 0.5, "y": 0.5, "r": 0.05}
    guess, emb = classifier.with_embedding(classifier.classify, img, circle, None, False, local=LOCAL, clf=clf)
    assert np.array_equal(emb, vec(3))
    assert not any(isinstance(v, np.ndarray) for v in guess.values())
    assert guess["crop"] == "circle" and db.valid_json(json.dumps(guess))
    # the crop box or the whole frame is not a tight view; a probs-only classifier has no embedding
    assert classifier.with_embedding(classifier.classify, img, None, (0.25, 0.25, 0.5, 0.5), False,
                                     local=LOCAL, clf=clf)[1] is None
    assert classifier.with_embedding(classifier.classify, img, circle, None, False, local=LOCAL,
                                     clf=FakeClassifier())[1] is None
    # nothing leaks to the next caller
    assert classifier.with_embedding(lambda: None)[1] is None
    clf.fail = True
    assert classifier.with_embedding(classifier.run, cv2.imencode(".jpg", img)[1].tobytes(), circle, None, False,
                                     )[1] is None


def test_tight_check_hands_its_embedding_out(monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER", "gate")
    img = np.full((1080, 1920, 3), 100, np.uint8)
    jpeg = cv2.imencode(".jpg", img)[1].tobytes()
    monkeypatch.setattr(classifier, "load", lambda: LookClassifier(probs=(0.9, 0.05, 0.02, 0, 0, 0.03, 0)))
    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)
    (guess, pick), emb = classifier.with_embedding(classifier.tight_check, jpeg, MOTION_CIRCLE, None, False,
                                                   {"Dark-eyed Junco": "bird"})
    assert pick == ("Dark-eyed Junco", "bird") and guess["crop"] == "motion" and np.array_equal(emb, vec(3))


# --- the watcher ----------------------------------------------------------------------------------------------

@pytest.fixture
def prints(gate_yard, monkeypatch, conn):
    """gate_yard with the real presence.store (so there is a row to attach to) and a classify that leaves an
    embedding the way the real one does: vec(1) for the motion circle's square, vec(2) for Claude's circle on the
    photo, vec(4) for Haiku's circle."""
    monkeypatch.setattr(watch.presence, "store", real_store)
    gate_yard["embed"] = {"motion": vec(1), "claude": vec(2), "haiku": vec(4)}
    gate_yard["claude_circle"] = {"x": 0.3, "y": 0.4, "r": 0.05}
    gate_yard["haiku_circle"] = {"x": 0.31, "y": 0.4, "r": 0.05}
    gate_yard["answer"] = {"escalated": False, "local_check": False, "circle": True}
    gate_yard["classify_calls"] = []

    def fake_classify(img, circle, box, night, local=None, clf=None):
        which = ("motion" if circle == MOTION_CIRCLE else "haiku" if circle == gate_yard["haiku_circle"]
                 else "claude" if circle else None)
        gate_yard["classify_calls"].append(which)
        if which == "motion":
            guess = gate_yard["tight"]
        else:
            guess = sure("Bushtit", 0.5)
        if guess is None:
            return None
        if which and gate_yard["embed"][which] is not None:
            classifier._mine.embedding = gate_yard["embed"][which]
        return {**guess, "crop": "motion" if which == "motion" else "circle" if which else "box"}

    monkeypatch.setattr(classifier, "classify", fake_classify)

    def fake_identify(*args, **kwargs):
        from birdbrain.identify import Result, Sighting
        gate_yard["asked"].append(kwargs.get("night"))
        a = gate_yard["answer"]
        raw = {}
        if a["local_check"] and kwargs.get("local_check"):
            guess = kwargs["local_check"](gate_yard["haiku_circle"])
            if guess:
                raw["classifier"] = guess
        if a["circle"]:
            raw["circle"] = gate_yard["claude_circle"]
        return Result(Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes="", circle=None,
                               also=[]), "claude-sonnet-5-5" if a["escalated"] else "claude-haiku-5-5",
                      a["escalated"], False, raw)

    monkeypatch.setattr(watch.identify, "identify", fake_identify)
    return gate_yard


def test_the_tight_look_that_names_the_visit_keeps_its_fingerprint(prints, conn):
    assert watch.handle_event() == "bird" and not prints["asked"]
    row = stored(conn, last_id(conn))
    assert np.allclose(decode(row["tight_vec"]), vec(1), atol=1e-3) and row["tight_view"] == classifier.TIGHT_VIEW
    assert row["circle_vec"] is None and row["circle_model"] is None
    assert np.allclose(decode(row["vec"]), one_hot(0, 0.01), atol=1e-3)  # the wide embedding is as before


def test_an_unsure_tight_look_and_claudes_circle_keep_both(prints, conn):
    prints["tight"] = sure("Dark-eyed Junco", 0.9)  # under the tight bar: Claude is asked
    assert watch.handle_event() == "bird" and prints["asked"] == [False]
    row = stored(conn, last_id(conn))
    assert np.allclose(decode(row["tight_vec"]), vec(1), atol=1e-3)
    assert np.allclose(decode(row["circle_vec"]), vec(2), atol=1e-3) and row["circle_model"] == "haiku"
    assert row["circle_view"] == classifier.TIGHT_VIEW
    raw = json.loads(conn.execute("SELECT raw_json FROM visits WHERE id = ?", (last_id(conn),)).fetchone()[0])
    assert raw["calls"]["classifier"]["top"][0]["name"] == "Bushtit"  # the stored guess is what it was
    assert "vec" not in json.dumps(raw["calls"]) and "tight_vec" not in json.dumps(raw)


def test_sonnets_circle_look_replaces_haikus(prints, conn):
    prints["tight"] = None  # the tight look has nothing to say
    prints["answer"] = {"escalated": True, "local_check": True, "circle": True}
    assert watch.handle_event() == "bird"
    assert prints["classify_calls"][-2:] == ["haiku", "claude"]
    row = stored(conn, last_id(conn))
    assert row["tight_vec"] is None  # a look that gave no guess keeps no fingerprint
    assert np.allclose(decode(row["circle_vec"]), vec(2), atol=1e-3) and row["circle_model"] == "sonnet"


def test_haikus_circle_look_is_kept_when_it_is_the_stored_one(prints, conn):
    prints["tight"] = None
    prints["answer"] = {"escalated": False, "local_check": True, "circle": True}
    assert watch.handle_event() == "bird"
    assert prints["classify_calls"].count("claude") == 0  # no second look: the stored guess is Haiku's circle
    row = stored(conn, last_id(conn))
    assert np.allclose(decode(row["circle_vec"]), vec(4), atol=1e-3) and row["circle_model"] == "haiku"


def test_no_circle_from_claude_means_no_circle_fingerprint(prints, conn):
    prints["tight"] = None
    prints["answer"] = {"escalated": False, "local_check": False, "circle": False}
    assert watch.handle_event() == "bird"
    row = stored(conn, last_id(conn))
    assert row["circle_vec"] is None and row["tight_vec"] is None


def test_a_failing_classifier_loses_nothing(prints, conn, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("model broke")
    monkeypatch.setattr(classifier, "classify", broken)
    assert watch.handle_event() == "bird" and prints["asked"] == [False]
    vid = last_id(conn)
    row = stored(conn, vid)
    assert row is not None and row["tight_vec"] is None and row["circle_vec"] is None
    assert conn.execute("SELECT species FROM visits WHERE id = ?", (vid,)).fetchone()[0] == "Bushtit"


def test_a_failing_fingerprint_write_never_loses_the_visit(prints, conn, monkeypatch):
    def boom(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(watch.presence, "store_tight", boom)
    assert watch.handle_event() == "bird"
    vid = last_id(conn)
    assert conn.execute("SELECT species FROM visits WHERE id = ?", (vid,)).fetchone()[0] == "Dark-eyed Junco"
    assert stored(conn, vid) is not None  # the wide embedding was kept


def test_nothing_is_kept_without_a_wide_embedding(prints, conn, monkeypatch):
    prints["emb"] = None  # the Pi animal check made none
    prints["tight"] = sure("Dark-eyed Junco", 0.9)
    assert watch.handle_event() == "bird"
    assert stored(conn, last_id(conn)) is None


# --- the backfill ---------------------------------------------------------------------------------------------

def test_the_backfill_embeds_claudes_circle_for_older_visits(conn, add_visit, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path)
    cv2.imwrite(str(tmp_path / "a.jpg"), np.full((1080, 1920, 3), 90, np.uint8))
    circle = {"x": 0.5, "y": 0.5, "r": 0.05}
    esc = add_visit(image_path="a.jpg", escalated=1, raw_json=json.dumps({"calls": {"circle": circle}}))
    hai = add_visit(image_path="a.jpg", raw_json=json.dumps({"calls": {"circle": circle}}))
    nocircle = add_visit(image_path="a.jpg", raw_json=json.dumps({"calls": {}}))
    norow = add_visit(image_path="a.jpg", raw_json=json.dumps({"calls": {"circle": circle}}))
    for vid in (esc, hai, nocircle):
        real_store(conn, vid, vec(0))

    class Encoder:
        def embed(self, bgr):
            return vec(5)

    lines = []
    assert classifier.backfill_circle_embeddings(clf=Encoder(), say=lines.append) == 2
    assert lines == ["2/2 looked at, 2 saved"]
    assert stored(conn, esc)["circle_model"] == "sonnet" and stored(conn, hai)["circle_model"] == "haiku"
    assert stored(conn, nocircle)["circle_vec"] is None and stored(conn, norow) is None
    assert classifier.backfill_circle_embeddings(clf=Encoder(), say=lines.append) == 0  # nothing left to do
