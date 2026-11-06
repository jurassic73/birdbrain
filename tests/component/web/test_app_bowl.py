"""Component tests of the web app through TestClient on a temp database: bowl."""

import json
import re

import numpy as np
import pytest

from helpers.presence_bowl import CROP, DIM, GROUND_CROP, LIVE, RING, frame
from birdbrain import classifier, db, presence
from birdbrain.web import app as webapp


def test_the_visit_photo_draws_the_pi_checks_view(client, add_visit):
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": LIVE}))
    html = client.get(f"/visit/{vid}").text
    x, y, w, h = classifier.presence_view_box(LIVE, "rim")
    want = f'x="{round(x * 160, 2)}" y="{round(y * 90, 2)}" width="{round(w * 160, 2)}" height="{round(h * 90, 2)}"'
    assert f'class="vz-crop"' in html and want in html
    cx, cy, cw, ch = CROP
    assert f'x="{round(cx * 160, 2)}" y="{round(cy * 90, 2)}"' not in html
    assert "a square around the whole bowl" in html
    vid = add_visit(visitor="none", species=None, raw_json=json.dumps({"trigger": "ground", "zones": LIVE}))
    html = client.get(f"/visit/{vid}").text
    gx, gy, gw, gh = GROUND_CROP
    assert f'x="{round(gx * 160, 2)}" y="{round(gy * 90, 2)}" width="{round(gw * 160, 2)}"' in html


@pytest.fixture
def client(conn):
    from fastapi.testclient import TestClient
    return TestClient(webapp.app)


BAD_RINGS = [RING[:5], [0.5, None, 0.1, 0.1, 0.08, 0.08], [0.5, float("nan"), 0.1, 0.1, 0.08, 0.08], "ring",
             [0.5, 0.5, 0, 0.1, 0.08, 0.08], [1.5, 0.5, 0.1, 0.1, 0.08, 0.08], list(RING) + [0] * 5]


@pytest.mark.parametrize("ring", BAD_RINGS)
def test_a_malformed_ring_is_no_ring_everywhere(ring, conn, add_visit, client):
    z = {"ring": ring, "crop": list(CROP)}
    assert classifier.usable_ring(ring) is None
    assert classifier.view_kind(z, "rim") == "padded" and classifier.presence_square((90, 160), z) is None
    assert classifier.presence_view_box(z, "rim") == CROP
    assert classifier.presence_patch(frame(90, 160), z, "rim")[1] == "padded"
    # Stored with a visit, it doesn't break retraining for everything else, or the visit's page.
    raw = {"zones": z, "trigger": "rim", "calls": {"circle": {"x": 0.47, "y": 0.45, "r": 0.03}}}
    vid = add_visit(raw_json=json.dumps(raw))
    presence.store(conn, vid, np.ones(DIM))
    assert presence.view_box(raw) == CROP
    assert vid in presence.examples(conn)["ids"]
    page = client.get(f"/visit/{vid}")
    assert page.status_code == 200
    x, y, w, h = CROP
    assert f'x="{round(x * 160, 2)}" y="{round(y * 90, 2)}"' in page.text and "Dashed box" in page.text


def test_stored_json_no_sqlite_can_read_breaks_nothing(conn, add_visit, client):
    """A visit whose stored JSON isn't JSON (here truncated, so no SQLite version reads it; a bare NaN from
    json.dumps is the same on the Pi's SQLite 3.40) must not raise "malformed JSON" out of the SQL json functions,
    which would take retraining and the visit page down for every visit, not just this one."""
    good = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": LIVE}))
    presence.store(conn, good, np.ones(DIM))
    bad = add_visit(species="Bushtit", also='[{"visitor": "raccoon"', raw_json='{"zones": {"ring": [0.5, ',
                    marks='[{"x": 0.5, "y": 0.5, "r": 0.1, "frame": 2')
    presence.store(conn, bad, np.ones(DIM))
    ex = presence.examples(conn)
    assert good in ex["ids"] and bad in ex["ids"]  # its photo still teaches "animal": Claude's crop showed it
    assert presence.retrain(conn)["key"] == presence.label_key(conn)
    assert presence.pending_picks(conn) == []  # no burst anyone can read, so nothing to look through
    assert presence.teaches(conn, bad)["used"]
    assert db.species_kinds(conn) == [("Bushtit", "bird"), ("Dark-eyed Junco", "bird")]
    for vid in (good, bad):
        assert client.get(f"/visit/{vid}").status_code == 200


def test_the_visit_page_draws_the_view_its_embedding_was_made_of(conn, add_visit, client):
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": LIVE}))

    def rect(box):
        x, y, w, h = box
        return f'x="{round(x * 160, 2)}" y="{round(y * 90, 2)}" width="{round(w * 160, 2)}" height="{round(h * 90, 2)}"'

    old = classifier.old_view_box(LIVE, "rim")
    assert abs(old[0] - 0.335) < 0.001 and abs(old[0] + old[2] - 0.615) < 0.001  # what the center cut kept
    presence.store(conn, vid, np.ones(DIM), view=None)
    html = client.get(f"/visit/{vid}").text
    assert rect(old) in html and "the middle square of the crop box" in html
    presence.store(conn, vid, np.ones(DIM))  # re-embedded
    html = client.get(f"/visit/{vid}").text
    assert rect(classifier.presence_view_box(LIVE, "rim")) in html and rect(old) not in html
    assert "a square around the whole bowl" in html and "Claude sees the crop box" in html
    # A ground visit of the old view: the middle of the ground crop; of the current view the ground crop itself.
    gid = add_visit(visitor="none", species=None, raw_json=json.dumps({"trigger": "ground", "zones": LIVE}))
    presence.store(conn, gid, np.ones(DIM), view=1)
    html = client.get(f"/visit/{gid}").text
    assert rect(classifier.old_view_box(LIVE, "ground")) in html and "the middle square of the ground crop" in html
    assert "Claude sees the ground crop" in html


def test_the_tip_names_the_view_for_no_ring_and_no_trigger(conn, add_visit, client):
    vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": {"crop": list(CROP)}}))
    html = client.get(f"/visit/{vid}").text
    assert "looked at, the crop box, so" in html and "whole bowl" not in html
    vid = add_visit(raw_json=json.dumps({"zones": LIVE}))  # no trigger saved: a birdbath view
    assert "looked at, a square around the whole bowl, so" in client.get(f"/visit/{vid}").text


def dashed(html) -> str:
    rects = re.findall(r'<rect class="vz-crop"[^>]*? (x="[\d.]+" y="[\d.]+" width="[\d.]+" height="[\d.]+")', html)
    assert len(rects) == 1
    return rects[0]


def rect_of(box) -> str:
    x, y, w, h = box
    return f'x="{round(x * 160, 2)}" y="{round(y * 90, 2)}" width="{round(w * 160, 2)}" height="{round(h * 90, 2)}"'


def test_the_dashed_box_goes_by_the_verdicts_view_first_then_the_embedding(conn, add_visit, client):
    new, old = rect_of(classifier.presence_view_box(LIVE, "rim")), rect_of(classifier.old_view_box(LIVE, "rim"))
    assert new != old

    def page(calls, view="none"):
        vid = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": LIVE, "calls": calls}))
        if view != "none":
            presence.store(conn, vid, np.ones(DIM), view=view)
        return client.get(f"/visit/{vid}").text

    named = {"presence": {"model": presence.MODEL_NAME, "score": 0.9, "threshold": 0.4, "empty": False, "view": "bowl"}}
    unnamed = {"presence": {"model": presence.MODEL_NAME, "score": 0.9, "threshold": 0.4, "empty": False}}
    # 1. The verdict names its view: the new box, whatever the embedding (or none, as on the mirror).
    assert dashed(page(named)) == new and dashed(page(named, view=None)) == new
    # 2. A verdict from before views had names: the old square (it was made on it), even after a redo.
    html = page(unnamed, view=classifier.PRESENCE_VIEW)
    assert dashed(html) == old and "the middle square of the crop box" in html
    html = page(unnamed)  # the mirror: no embeddings, the verdict kept in raw_json
    assert dashed(html) == old and "the middle square of the crop box" in html
    # 3. No verdict: the embedding's view, old or current.
    html = page({}, view=None)
    assert dashed(html) == old and "the middle square of the crop box" in html
    assert dashed(page({}, view=classifier.PRESENCE_VIEW)) == new
    # 4. Nothing about the Pi animal check at all: the new box.
    html = page({})
    assert dashed(html) == new and "a square around the whole bowl" in html


def test_the_mirror_draws_the_same_box_as_the_pi_for_new_and_old_visits(conn, add_visit, client):
    named = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": LIVE, "calls": {
        "presence": {"model": presence.MODEL_NAME, "score": 0.9, "threshold": 0.4, "empty": False, "view": "bowl"}}}))
    unnamed = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": LIVE, "calls": {
        "presence": {"model": presence.MODEL_NAME, "score": 0.9, "threshold": 0.4, "empty": False}}}))
    redone = add_visit(raw_json=json.dumps({"trigger": "rim", "zones": LIVE, "calls": {
        "presence": {"model": presence.MODEL_NAME, "score": 0.9, "threshold": 0.4, "empty": False}}}))
    presence.store(conn, named, np.ones(DIM))
    presence.store(conn, unnamed, np.ones(DIM), view=None)
    presence.store(conn, redone, np.ones(DIM))  # a verdict from before views had names, re-embedded since
    visits = (named, unnamed, redone)
    on_pi = {vid: dashed(client.get(f"/visit/{vid}").text) for vid in visits}
    assert on_pi[redone] == on_pi[unnamed] == rect_of(classifier.old_view_box(LIVE, "rim"))
    conn.execute("DELETE FROM embeddings")  # what mirror.trim does
    conn.commit()
    assert {vid: dashed(client.get(f"/visit/{vid}").text) for vid in visits} == on_pi
