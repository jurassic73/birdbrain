import io
import json
from datetime import datetime

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from birdbrain import classifier, config, db, images, watch
from birdbrain.classifier import fetch_place_species as real_fetch_place_species
from birdbrain.web import app as webapp

LABELS = [("Junco hyemalis", "Dark-eyed Junco", True), ("Junco hyemalis", "Oregon Junco", False),
          ("Haemorhous mexicanus", "house finch", True), ("Haemorhous purpureus", "Purple Finch", True),
          ("Picoides pubescens", "Downy Woodpecker", True), ("Garrulus glandarius", "Eurasian Jay", True),
          ("background", "", True)]
LOCAL = {"Junco hyemalis": "Dark-eyed Junco", "Haemorhous mexicanus": "House Finch",
         "Dryobates pubescens": "Downy Woodpecker"}


class FakeClassifier:
    labels = LABELS

    def __init__(self, probs=(0.2, 0.3, 0.1, 0.05, 0.05, 0.3, 0.0), fail=False):
        self._probs, self.fail, self.shapes = np.array(probs, np.float32), fail, []

    def probs(self, bgr):
        if self.fail:
            raise RuntimeError("model broke")
        self.shapes.append(bgr.shape)
        return self._probs


def test_rank_keeps_local_species_and_renormalizes():
    r = classifier.rank(np.array([0.2, 0.3, 0.1, 0.05, 0.05, 0.3, 0.0]), LABELS, LOCAL)
    # Junco 0.2 + Oregon 0.3; the Eurasian Jay and Purple Finch aren't local; Downy matched by its common name
    # although its genus was renamed; eBird spelling for "house finch".
    assert [g["name"] for g in r["top"]] == ["Dark-eyed Junco", "House Finch", "Downy Woodpecker"]
    assert [g["score"] for g in r["top"]] == [pytest.approx(0.5 / 0.65, abs=1e-3),
                                              pytest.approx(0.1 / 0.65, abs=1e-3),
                                              pytest.approx(0.05 / 0.65, abs=1e-3)]
    assert r["top"][2]["scientific"] == "Picoides pubescens"
    assert r["masked"] and r["species"] == 3 and r["local_share"] == pytest.approx(0.65, abs=1e-3)


def test_rank_without_a_local_list_uses_every_species():
    r = classifier.rank(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1]), LABELS, None)
    assert r["top"][0]["name"] == "Eurasian Jay" and r["top"][0]["score"] == 1.0
    assert not r["masked"] and r["local_share"] is None and r["background"] == pytest.approx(0.1)


def test_crop_is_a_square_around_the_circle_else_the_box():
    img = np.zeros((1080, 1920, 3), np.uint8)
    patch, used = classifier.crop_for(img, {"x": 0.5, "y": 0.5, "r": 0.05}, (0.25, 0.25, 0.5, 0.5))
    side = round(2 * 0.05 * 1920 * classifier.CIRCLE_MARGIN)
    assert used == "circle" and patch.shape[:2] == (side, side)
    patch, used = classifier.crop_for(img, {"x": 0.99, "y": 0.01, "r": 0.05}, None)  # near a corner: slides in
    assert used == "circle" and patch.shape[:2] == (side, side)
    patch, used = classifier.crop_for(img, None, (0.25, 0.25, 0.5, 0.5))
    assert used == "box" and patch.shape[:2] == (540, 960)
    assert classifier.crop_for(img, None, None)[1] == "frame"


def test_place_species_are_fetched_page_by_page_and_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(classifier.time, "sleep", lambda s: None)
    pages = []

    def get(params):
        pages.append(params["page"])
        assert params["place_id"] == "502" and params["iconic_taxa"] == "Aves"
        taxa = [{"count": 9, "taxon": {"name": "Junco hyemalis", "preferred_common_name": "Dark-eyed Junco"}}]
        if params["page"] == 2:
            taxa = [{"count": 1, "taxon": {"name": "Junco hyemalis oreganus", "preferred_common_name": "Oregon Junco"}},
                    {"count": 1, "taxon": {"name": "Psaltriparus minimus", "preferred_common_name": "Bushtit"}}]
        return {"total_results": 3, "per_page": 2, "results": taxa}

    species = real_fetch_place_species("502", get=get)
    assert pages == [1, 2] and [s["name"] for s in species] == ["Junco hyemalis", "Junco hyemalis", "Psaltriparus minimus"]

    monkeypatch.setattr(classifier, "fetch_place_species", lambda place, taxon="Aves": species)
    local = classifier.place_species("502")
    assert local["Psaltriparus minimus"] == "Bushtit" and (tmp_path / "inat_birds_502.json").exists()
    # Offline later: the cached list still works, stale or not.
    monkeypatch.setattr(classifier, "PLACE_MAX_AGE", classifier.timedelta(seconds=-1))
    monkeypatch.setattr(classifier, "fetch_place_species", lambda place, taxon="Aves": 1 / 0)
    assert classifier.place_species("502") == local
    assert classifier.place_species("") is None
    assert classifier.place_species("999") is None  # never fetched and offline: no limit


def test_watcher_stores_the_guess_without_changing_claudes_label(monkeypatch, conn):
    burst = [cv2.imencode(".jpg", np.full((1080, 1920, 3), (40, 120, 200), np.uint8))[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "off")

    def fake_identify(*args, **kwargs):
        from birdbrain.identify import Result, Sighting
        return Result(Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes=""),
                      "claude-haiku-5-5", False, False, {"circle": {"x": 0.5, "y": 0.5, "r": 0.03, "claude": True}})

    monkeypatch.setattr(watch.identify, "identify", fake_identify)
    fake = FakeClassifier()
    monkeypatch.setattr(classifier, "load", lambda: fake)
    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)
    assert watch.handle_event() == "bird"
    row = conn.execute("SELECT species, raw_json FROM visits").fetchone()
    guess = json.loads(row["raw_json"])["calls"]["classifier"]
    assert row["species"] == "Bushtit"  # Claude's label stands
    assert guess["crop"] == "circle" and guess["top"][0]["name"] == "Dark-eyed Junco" and len(guess["top"]) == 3
    assert guess["model"] == classifier.MODEL_NAME and "seconds" in guess and guess["night"] is False

    fake.fail = True  # fail soft: the visit is stored without a guess
    assert watch.handle_event() == "bird"
    raws = [json.loads(r[0]) for r in conn.execute("SELECT raw_json FROM visits ORDER BY id")]
    assert "classifier" not in raws[1]["calls"]


def test_off_means_no_guess(monkeypatch):
    monkeypatch.setattr(config, "CLASSIFIER", "off")
    monkeypatch.setattr(classifier, "load", lambda: 1 / 0)
    assert classifier.run(b"not even a jpeg", None, None, False) is None


def test_a_failed_download_is_not_retried_on_every_visit(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    tries = []
    monkeypatch.setattr(classifier, "download_model", lambda path: tries.append(path) or 1 / 0)
    for _ in range(3):
        assert classifier.run(cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes(), None, None, False) is None
    assert len(tries) == 1


def guess(*names, night=False):
    return {"model": classifier.MODEL_NAME, "crop": "circle", "night": night, "seconds": 0.05,
            "top": [{"name": n, "scientific": "", "score": 0.5} for n in names]}


def raw(*names, night=False):
    return json.dumps({"calls": {"classifier": guess(*names, night=night)}})


def test_report_scores_against_the_final_label(conn, add_visit):
    add_visit(species="Dark-eyed Junco", raw_json=raw("Dark-eyed Junco", "Bushtit"))            # top 1
    add_visit(species="Bushtit", raw_json=raw("Dark-eyed Junco", "Bushtit"))                     # top 3
    v = add_visit(species="Bushtit", raw_json=raw("House Finch", "Dark-eyed Junco"))
    db.correct_visit(conn, v, "bird", "house finch", None)  # the owner's label wins (key ignores case)
    add_visit(species="Bushtit", is_night=1, raw_json=raw("Song Sparrow", night=True))           # miss
    add_visit(species=None, raw_json=raw("Bushtit"))                                             # "Bird": no species
    add_visit(visitor="squirrel", species="Eastern Gray Squirrel", raw_json=raw("Eastern Gray Squirrel"))  # mammals too
    add_visit(visitor="raccoon", species=None, raw_json=raw("Common Raccoon"))                   # "Raccoon" matches
    add_visit(visitor="squirrel", species=None, raw_json=raw("Douglas's Squirrel"))              # "Squirrel": no species
    add_visit(species="Bushtit")                                                                 # never scored
    r = db.classifier_report(conn)
    assert (r["checked"], r["top1"], r["top3"], r["night"], r["night_top1"]) == (6, 4, 5, 1, 0)
    assert r["seconds"] == 0.05
    bushtit = next(s for s in r["species"] if s["label"] == "Bushtit")
    assert (bushtit["visits"], bushtit["top1"], bushtit["top3"]) == (2, 0, 1)
    assert [m["guess"] for m in r["misses"]] == ["Song Sparrow", "Dark-eyed Junco"]


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


def test_visit_page_shows_the_guess_as_shadow(client, add_visit):
    vid = add_visit(species="Bushtit", raw_json=json.dumps({"calls": {
        "haiku": {"answer": {"visitor": "bird", "species": "Bushtit", "confidence": "high"}},
        "classifier": guess("Bushtit", "Dark-eyed Junco")}}))
    html = client.get(f"/visit/{vid}").text
    assert "Local classifier" in html and "Bushtit (50%), Dark-eyed Junco (50%)" in html
    assert 'class="badge shadow" data-tip="Shadow:' in html
    assert "asked Sonnet" not in html  # the classifier step after Haiku isn't a second look


def test_stats_page_shows_the_scorecard(client, add_visit):
    assert "No bird visits scored yet" in client.get("/stats").text
    add_visit(species="Bushtit", raw_json=raw("Bushtit"))
    html = client.get("/stats").text
    assert 'id="classifier"' in html and "100%</span> first guess right (1/1)" in html


def test_backfill_scores_stored_visits(conn, add_visit):
    def stored(**fields):
        buf = io.BytesIO()
        Image.new("RGB", (1920, 1080), (90, 120, 200)).save(buf, "JPEG")
        image_rel, thumb_rel = images.store_frame(buf.getvalue(), datetime.now())
        return add_visit(image_path=image_rel, thumb_path=thumb_rel, **fields)

    old = stored(raw_json=json.dumps({"calls": {"circle": {"x": 0.5, "y": 0.5, "r": 0.02}}}))
    owner = stored()
    db.correct_visit(conn, owner, "bird", "Bushtit", None, marks=[{"x": 0.2, "y": 0.3, "r": 0.04}])
    stored(visitor="none", species=None)  # false triggers aren't scored
    done = stored(raw_json=raw("Bushtit"))
    fake = FakeClassifier()
    assert classifier.backfill(limit=1, clf=fake, local=LOCAL) == 1  # newest unscored first
    assert classifier.backfill(clf=fake, local=LOCAL) == 1
    assert classifier.backfill(clf=fake, local=LOCAL) == 0
    calls = {r["id"]: json.loads(r["raw_json"] or "{}").get("calls", {}) for r in conn.execute("SELECT id, raw_json FROM visits")}
    assert calls[old]["classifier"]["crop"] == "circle" and calls[old]["classifier"]["backfilled"]
    assert calls[owner]["classifier"]["crop"] == "circle"
    assert "backfilled" not in calls[done]["classifier"]  # left alone without --redo
    assert fake.shapes[0][:2] == (round(2 * 0.04 * 1920 * classifier.CIRCLE_MARGIN),) * 2  # the owner's circle
    assert classifier.backfill(redo=True, clf=fake, local=LOCAL) == 3


def test_a_circle_is_classified_at_each_margin_and_averaged(monkeypatch):
    monkeypatch.setattr(classifier, "CIRCLE_MARGINS", (1.5, 3.0))
    img = np.zeros((1080, 1920, 3), np.uint8)
    fake = FakeClassifier()
    r = classifier.classify(img, {"x": 0.5, "y": 0.5, "r": 0.03}, None, False, local=LOCAL, clf=fake)
    assert [s[0] for s in fake.shapes] == [round(2 * 0.03 * 1920 * m) for m in classifier.CIRCLE_MARGINS]
    assert r["crop"] == "circle" and r["top"]
    fake.shapes.clear()
    classifier.classify(img, None, (0.25, 0.25, 0.5, 0.5), False, local=LOCAL, clf=fake)
    assert len(fake.shapes) == 1  # the crop box is classified once


def test_input_is_a_normalized_center_crop():
    x = classifier.to_input(np.full((300, 500, 3), 255, np.uint8))
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx((1 - classifier.MEAN[0]) / classifier.STD[0], abs=1e-4)


def test_probs_are_a_softmax_over_the_species_embeddings(monkeypatch, tmp_path):
    names = [["Junco hyemalis", "Dark-eyed Junco"], ["Haemorhous mexicanus", "House Finch"]]
    np.save(tmp_path / "bioclip2_ebird_text.npy", np.array([[1, 0], [0, 1]], np.float16))
    (tmp_path / "bioclip2_ebird_names.json").write_text(json.dumps(names))
    np.save(tmp_path / "bioclip2_mammals_text.npy", np.array([[-1, 0]], np.float16))
    (tmp_path / "bioclip2_mammals_names.json").write_text(json.dumps([["Sciurus carolinensis", "Eastern Gray Squirrel"]]))

    class Session:
        def __init__(self, *a):
            pass

        def run(self, out, feed):
            assert feed["image"].shape == (1, 3, 224, 224)
            return [np.array([[0.6, 0.8]], np.float32)]

    import onnxruntime
    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    clf = classifier.Classifier(tmp_path)
    p = clf.probs(np.zeros((100, 100, 3), np.uint8))
    assert p.shape == (3,) and p.sum() == pytest.approx(1) and p[1] > 0.99  # 100 x (0.8 - 0.6) apart
    assert clf.labels[2] == ("Sciurus carolinensis", "Eastern Gray Squirrel", True)
    r = classifier.rank(p, clf.labels, {"Junco hyemalis": "Dark-eyed Junco", "Haemorhous mexicanus": "House Finch"})
    assert r["top"][0]["name"] == "House Finch" and r["background"] == 0


class FakeTextClassifier:
    """Embeddings as one-hot vectors over LABELS, so the species scores are known exactly."""
    labels = LABELS
    text = np.eye(len(LABELS), dtype=np.float32)


def one_hot(i, k=1.0):
    v = np.zeros(len(LABELS), np.float32)
    v[i] = k
    return v


def test_gate_names_only_sure_daytime_guesses_of_settled_species():
    clf = FakeTextClassifier()
    sure = classifier.from_embedding(one_hot(0), False, local=LOCAL, clf=clf)
    assert sure["top"][0]["name"] == "Dark-eyed Junco" and sure["top"][0]["score"] > 0.99 and sure["crop"] == "box"
    settled = {"Dark-eyed Junco": "bird"}
    assert classifier.gate_pick(sure, settled, night=False) == ("Dark-eyed Junco", "bird")
    assert classifier.gate_pick(sure, settled, night=True) is None            # never at night (IR unseen)
    assert classifier.gate_pick(sure, {}, night=False) is None                # not settled in the yard yet
    unsure = classifier.from_embedding(one_hot(0, 0.01), False, local=LOCAL, clf=clf)
    assert classifier.gate_pick(unsure, settled, night=False) is None         # spread over every species
    foreign = dict(sure, local_share=0.2)
    assert classifier.gate_pick(foreign, settled, night=False) is None        # mostly a species from elsewhere


def test_settled_species_ignores_visits_the_gate_named_itself(conn, add_visit):
    for _ in range(3):
        add_visit(species="Dark-eyed Junco")
        add_visit(visitor="squirrel", species="Eastern Gray Squirrel")
        add_visit(species="House Finch", model="bioclip-2")
    add_visit(species="Bushtit", confidence="medium")
    assert db.settled_species(conn) == {"Dark-eyed Junco": "bird", "Eastern Gray Squirrel": "squirrel"}


def test_watcher_skips_claude_when_the_gate_is_sure(monkeypatch, conn, add_visit):
    for _ in range(3):
        add_visit(species="Dark-eyed Junco")
    burst = [cv2.imencode(".jpg", np.full((1080, 1920, 3), (40, 120, 200), np.uint8))[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    monkeypatch.setattr(watch, "presence_mode", lambda: "gate")
    monkeypatch.setattr(config, "CLASSIFIER", "gate")
    emb = {"v": one_hot(0)}
    monkeypatch.setattr(watch.classifier, "box_embedding", lambda jpeg, box: emb["v"])
    monkeypatch.setattr(watch.presence, "check", lambda conn, e: {"model": "bioclip-presence", "score": 0.9,
                                                                  "threshold": 0.4, "empty": False})
    monkeypatch.setattr(watch.presence, "store", lambda conn, vid, e: None)
    monkeypatch.setattr(classifier, "load", lambda: FakeTextClassifier())
    monkeypatch.setattr(classifier, "local_species", lambda: LOCAL)
    asked = []

    def fake_identify(*args, **kwargs):
        from birdbrain.identify import Result, Sighting
        asked.append(1)
        return Result(Sighting(visitor="bird", species="Bushtit", count=2, confidence="high", notes=""),
                      "claude-haiku-5-5", False, False, {})

    monkeypatch.setattr(watch.identify, "identify", fake_identify)
    monkeypatch.setattr(classifier, "run", lambda *a: None)
    assert watch.handle_event() == "bird"
    row = conn.execute("SELECT species, model, confidence, raw_json FROM visits ORDER BY id DESC").fetchone()
    calls = json.loads(row["raw_json"])["calls"]
    assert not asked and row["species"] == "Dark-eyed Junco" and row["model"] == "bioclip-2"
    assert calls["skipped_claude"] and calls["settled_by"] == "classifier" and calls["presence"]["score"] == 0.9

    emb["v"] = one_hot(0, 0.01)  # unsure: Claude names it
    assert watch.handle_event() == "bird" and asked
    assert conn.execute("SELECT species FROM visits ORDER BY id DESC").fetchone()[0] == "Bushtit"


def test_visit_page_says_the_pi_named_it(client, add_visit):
    vid = add_visit(model="bioclip-2", raw_json=json.dumps({"calls": {
        "classifier": guess("Dark-eyed Junco"), "skipped_claude": True, "settled_by": "classifier"}}))
    html = client.get(f"/visit/{vid}").text
    assert "Claude wasn't asked" in html and 'data-tip="Pi named:' in html
