"""The Hailo detector card on Stats through TestClient: admin only, hidden without the detector's database, cheap
to skip when that database is unreadable, and its snapshot route serves thumbnails of the detector's files only."""
import io
import os
from datetime import timedelta

import pytest
from PIL import Image

from birdbrain import config
from birdbrain.hailo import store
from helpers.clock import FROZEN_NOW

T = FROZEN_NOW


def _iso(delta_s: float) -> str:
    return (T + timedelta(seconds=delta_s)).isoformat(timespec="seconds")


def _jpeg(width=640, height=360) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (40, 90, 60)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def shadow_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HAILO_SHADOW_DB", tmp_path / "hailo_shadow.db")
    monkeypatch.setattr(config, "HAILO_SNAPSHOT_DIR", tmp_path / "snaps")
    monkeypatch.setattr(config, "HAILO_HEARTBEAT", tmp_path / "hb.json")
    return tmp_path


SNAP = "2026/11/10/115000_2_start.jpg"
SNAP_PRE = "2026/11/10/115000_2_pre2s.jpg"


@pytest.fixture
def seeded(shadow_paths, add_visit):
    """A detector database with one event beside a real visit (seen 20 s before the trigger), one beside a false
    trigger, one detector-only event with two snapshots, plus a fresh heartbeat; and visits the detector missed."""
    with store.session() as conn:
        store.insert_event(conn, started_at=_iso(-3600), ended_at=_iso(-3590), label="bird", score=0.8,
                           labels={"bird": 8}, animal=True, zone="bath", in_zone=True, frames=8, seconds=10.0)
        store.insert_event(conn, started_at=_iso(-600), ended_at=_iso(-590), label="bird", score=0.55,
                           labels={"bird": 12}, animal=True, zone="crop", in_zone=True, frames=12, seconds=10.0,
                           snapshots=[SNAP_PRE, SNAP, "2026/11/10/115000_2_end.jpg"], night=0)
        store.insert_event(conn, started_at=_iso(-1800), ended_at=_iso(-1795), label="cat", score=0.4,
                           labels={"cat": 5}, animal=True, zone="bath", in_zone=True, frames=5)
        store.insert_event(conn, started_at=_iso(-900), ended_at=_iso(-899), label="chair", score=0.9,
                           labels={"chair": 30}, animal=False, zone="elsewhere", in_zone=False, frames=30)
    for rel in (SNAP_PRE, SNAP):
        store.save_jpeg(rel, _jpeg())
    store.write_heartbeat({"at": _iso(-10), "fps": 15.0, "infer_ms": 11.8, "cpu_pct": 21.0, "chip_temp_c": 43.1,
                           "model": "yolov8s_h8", "view": "crop", "frames": 27690, "uptime_s": 1830})
    add_visit(captured_at=_iso(-3580), visitor="bird")                       # seen 20 s before the trigger
    add_visit(captured_at=_iso(-1790), visitor="none")                       # a false trigger the detector saw too
    add_visit(captured_at=_iso(-5000), visitor="none")                       # a false trigger it left alone
    add_visit(captured_at=_iso(-7000), visitor="squirrel", species=None)     # missed
    return shadow_paths


def card(html: str) -> str:
    assert 'id="hailo"' in html
    start = html.index('id="hailo"')
    return html[start:html.index("</section>", start)]


def test_the_card_is_left_out_without_the_detector_database(client, shadow_paths):
    html = client.get("/stats").text
    assert 'id="hailo"' not in html and "Hailo detector" not in html


def test_admin_sees_health_scorecard_classes_and_the_review_list(client, seeded):
    html = client.get("/stats").text
    c = card(html)
    assert "Running" in c and "15.0</b> fps" in c and "11.8</b> ms" in c and "43.1&deg;C" in c
    assert "yolov8s_h8" in c and "crop box" in c and "27,690" in c
    assert '<span class="big">3<small> / 4</small></span> animal-like events' in c
    assert "50%</span> of visits also seen (1/2)" in c
    assert "20 s</span> ahead of the trigger" in c
    assert "50%</span> of false triggers left alone (1/2)" in c
    assert "1</span> visits it missed" in c and "1</span> detector-only events" in c
    assert "<td>bird</td><td>bird</td>" in c and "<td>false trigger</td><td>cat</td>" in c
    assert f'src="/hailo/snapshot/{SNAP}"' in c and f'src="/hailo/snapshot/{SNAP_PRE}"' in c
    assert "115000_2_end.jpg" not in c  # the "after" shot is not part of the arrival strip
    assert "2 s before" in c and "first frame" in c
    assert 'class="score-range on"' in c and 'hx-get="/hailo/card?since=24h"' in c
    assert "stored photos" in c  # the offline scorecard in the collapsed More


def test_the_fragment_route_changes_the_window(client, seeded):
    r = client.get("/hailo/card?since=24h")
    assert r.status_code == 200 and 'data-window="24h"' in r.text
    assert 'aria-pressed="true"' in r.text.split("24 h")[0]  # the 24 h button is the pressed one
    assert client.get("/hailo/card?since=bogus").text.count('data-window="3d"') == 1


def test_visitors_never_see_the_card_or_its_routes(client, seeded, locked):
    html = client.get("/stats").text
    assert 'id="hailo"' not in html
    assert client.get("/hailo/card", follow_redirects=False).status_code == 303
    r = client.get(f"/hailo/snapshot/{SNAP}", follow_redirects=False)
    assert r.status_code == 303 and "image" not in r.headers.get("content-type", "")
    client.post("/login", data={"password": "s3cret!"})
    assert 'id="hailo"' in client.get("/stats").text
    r = client.get(f"/hailo/snapshot/{SNAP}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert "private" in r.headers["cache-control"]


def test_the_public_mirror_has_neither_card_nor_routes(client, seeded, monkeypatch):
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    assert 'id="hailo"' not in client.get("/stats").text
    assert client.get("/hailo/card").status_code == 404
    assert client.get(f"/hailo/snapshot/{SNAP}").status_code == 404


def test_snapshots_are_thumbnails_of_the_detectors_files_only(client, seeded):
    r = client.get(f"/hailo/snapshot/{SNAP}")
    img = Image.open(io.BytesIO(r.content))
    assert img.width <= 320 and img.height <= 180
    assert client.get("/hailo/snapshot/2026/11/10/115000_9_start.jpg").status_code == 404  # right shape, no file
    assert client.get("/hailo/snapshot/2026/11/10/115000_2_start.png").status_code == 404
    assert client.get("/hailo/snapshot/%2e%2e/%2e%2e/hb.json").status_code == 404
    assert client.get("/hailo/snapshot/2026/11/10/..%2f..%2f..%2fhb.json").status_code == 404
    # a symlink with the right name pointing out of the folder is refused too
    outside = seeded / "outside.jpg"
    outside.write_bytes(_jpeg(8, 8))
    link = config.HAILO_SNAPSHOT_DIR / "2026/11/10/115000_7_best.jpg"
    os.symlink(outside, link)
    assert client.get("/hailo/snapshot/2026/11/10/115000_7_best.jpg").status_code == 404


def test_an_empty_database_and_a_stale_heartbeat_still_render(client, shadow_paths):
    with store.session() as conn:
        conn.execute("SELECT 1")
    store.write_heartbeat({"at": _iso(-3600), "fps": 0.0, "model": "yolov8s_h8"})
    c = card(client.get("/stats").text)
    assert "Stale" in c and "60 min ago" in c
    assert '<span class="big">0<small> / 0</small></span>' in c and "--</span> of visits also seen (0/0)" in c
    assert "nothing to compare labels on" in c and "every animal-like event" in c


def test_no_heartbeat_at_all_says_not_running(client, shadow_paths):
    with store.session() as conn:
        conn.execute("SELECT 1")
    assert "Not running" in card(client.get("/stats").text)


def test_an_unreadable_detector_database_leaves_the_card_out_and_the_page_up(client, shadow_paths):
    config.HAILO_SHADOW_DB.write_bytes(b"this is not a database")
    r = client.get("/stats")
    assert r.status_code == 200 and 'id="hailo"' not in r.text
    assert client.get("/hailo/card").status_code == 404
