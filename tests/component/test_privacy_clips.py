"""Component tests of the person check on visit clips (birdbrain/privacy.py) and staging them for the mirror
(birdbrain/mirror.py): clip files on disk, a temp database, the detector faked."""

import json
import sqlite3

import cv2
import numpy as np
import pytest

from birdbrain import clips, config, db, mirror, privacy


def write_clip(rel: str, frames: int = 50, fps: int = 25, person_at: int | None = None) -> None:
    """A small real video: gray frames, white for half a second from `person_at` on (the fake detector's person)."""
    path = config.CLIPS_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 36))
    for i in range(frames):
        walking = person_at is not None and person_at <= i < person_at + fps // 2
        out.write(np.full((36, 64, 3), 255 if walking else 90, np.uint8))
    out.release()


@pytest.fixture
def detector(monkeypatch):
    """person_score faked: 0.9 on a white frame, 0.01 otherwise; counts the frames looked at."""
    seen = []

    def score(bgr):
        seen.append(1)
        return 0.9 if bgr.mean() > 200 else 0.01
    monkeypatch.setattr(privacy, "person_score", score)
    return seen


def test_a_clip_is_looked_at_twice_a_second_and_on_its_last_frame(detector):
    write_clip("2026/11/11/a.mp4", frames=50)  # 2 s at 25 fps
    assert privacy.score_clip(config.CLIPS_DIR / "2026/11/11/a.mp4") == 0.01
    assert len(detector) == 6  # every 12th frame (0, 12, 24, 36, 48) and the last one


@pytest.mark.parametrize("at", [0, 20, 49])  # first frame, between checked frames, the very last
def test_someone_in_view_for_half_a_second_is_seen(detector, at):
    rel = "2026/11/11/p.mp4"
    path = config.CLIPS_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 36))
    for i in range(50):
        walking = at <= i < at + 13  # about half a second in view
        out.write(np.full((36, 64, 3), 255 if walking else 90, np.uint8))
    out.release()
    assert privacy.score_clip(path) == 0.9


def test_a_clip_that_cannot_be_read_counts_as_unchecked(detector):
    bad = config.CLIPS_DIR / "2026/11/11/bad.mp4"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        privacy.score_clip(bad)


def mirror_copy(conn, add_visit, tmp_path, *clips_spec):
    """Visits with clean photos and the given clips, and the mirror's prepared copy of them."""
    ids = {}
    for name, person, extra in clips_spec:
        rel = extra.pop("clip_path", f"2026/11/11/{name}.mp4")
        if person is not None:
            write_clip(rel, person_at=10 if person else None)
        ids[name] = add_visit(image_path=f"2026/11/11/{name}.jpg", thumb_path=f"2026/11/11/{name}_t.jpg",
                              clip_path=rel, **extra)
    for vid in ids.values():  # the photos themselves: on disk and clean
        image = db.get_visit(conn, vid)["image_path"]
        (config.IMAGES_DIR / image).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(config.IMAGES_DIR / image), np.full((36, 64, 3), 90, np.uint8))
    conn.commit()
    stage = tmp_path / "stage"
    mirror.prepare(config.DB_PATH, stage, "America/Los_Angeles")
    names = {row[0]: row[1] for row in sqlite3.connect(stage / "birdbrain.db").execute("SELECT id, clip_path FROM visits")}
    return ids, names, stage


def test_only_clean_clips_go_to_the_mirror_and_their_visits_go_either_way(conn, add_visit, tmp_path, detector):
    ids, names, stage = mirror_copy(conn, add_visit, tmp_path,
                                    ("clean", False, {}), ("walker", True, {}), ("gone", None, {}))
    assert names[ids["clean"]] == "2026/11/11/clean.mp4"
    assert names[ids["walker"]] is None and names[ids["gone"]] is None  # the visits still go, without a clip
    staged = sorted(str(p.relative_to(stage / "clips")) for p in (stage / "clips").rglob("*.mp4"))
    assert staged == ["2026/11/11/clean.mp4"]
    assert (stage / "clips.txt").read_text() == "2026/11/11/clean.mp4\n"
    cache = json.loads(config.DATA_DIR.joinpath("privacy.json").read_text())
    assert cache["clips/2026/11/11/walker.mp4"] == 0.9 and cache["clips/2026/11/11/clean.mp4"] == 0.01
    assert db.get_visit(conn, ids["walker"])["clip_path"] == "2026/11/11/walker.mp4"  # the Pi keeps it


def test_make_clip_public_lets_a_flagged_clip_through(conn, add_visit, tmp_path, detector):
    write_clip("2026/11/11/w.mp4", person_at=10)
    vid = add_visit(image_path="2026/11/11/w.jpg", thumb_path="2026/11/11/w_t.jpg", clip_path="2026/11/11/w.mp4")
    db.set_public_ok(conn, vid, privacy.clip_override(None, "2026/11/11/w.mp4", 0.9))
    _, names, _ = mirror_copy(conn, add_visit, tmp_path)
    assert names[vid] == "2026/11/11/w.mp4"


def test_unchecked_clips_wait_when_the_time_runs_out(conn, add_visit, tmp_path, detector, monkeypatch):
    monkeypatch.setattr(privacy, "FILTER_BUDGET", 0.0)
    ids, names, stage = mirror_copy(conn, add_visit, tmp_path, ("later", False, {}))
    assert names[ids["later"]] is None and not list((stage / "clips").rglob("*.mp4"))
    assert "clips/2026/11/11/later.mp4" not in privacy.load_cache()


def test_a_clip_path_outside_the_folder_is_never_staged(conn, add_visit, tmp_path, detector):
    outside = config.DATA_DIR / "secret.mp4"
    outside.write_bytes(b"x" * 100)
    ids, names, stage = mirror_copy(conn, add_visit, tmp_path, ("evil", None, {"clip_path": "../secret.mp4"}))
    assert names[ids["evil"]] is None and not list((stage / "clips").rglob("*"))


def test_the_clips_folder_is_always_staged_so_the_mirror_empties_it(conn, add_visit, tmp_path, detector):
    _, _, stage = mirror_copy(conn, add_visit, tmp_path)
    assert (stage / "clips").is_dir() and (stage / "clips.txt").read_text() == ""


def test_the_clip_pill_reads_the_cache(add_visit, conn):
    rel = "2026/11/11/s.mp4"
    assert privacy.clip_status(rel) is None  # the filter never ran
    privacy.save_cache({})
    assert privacy.clip_status(rel) == "unchecked" and privacy.clip_status(None) is None
    privacy.save_cache({privacy.clip_key(rel): 0.9})
    assert privacy.clip_status(rel) == "person"
    cleared = json.dumps(privacy.clip_override(None, rel, 0.9))
    assert privacy.clip_status(rel, cleared) == "cleared"
    privacy.save_cache({privacy.clip_key(rel): 0.95})  # it scores higher on a re-check: held back again
    assert privacy.clip_status(rel, cleared) == "person"
    assert clips.inside(rel)
