"""Component tests of birdbrain/clips.py: recordings and clip files on disk, pruning on a temp database."""

import os
import subprocess
import time
from datetime import datetime, timedelta

import pytest

from helpers.clock import FROZEN_NOW
from birdbrain import clips, config, db, retention

VIDEO = b"\x00" * (clips.MIN_BYTES + 1)


class FakeFfmpeg:
    """The burst's ffmpeg as Recording sees it: done with an exit code, or still running until it is stopped."""

    def __init__(self, code: int = 0, hangs: bool = False) -> None:
        self.code, self.hangs, self.stopped = code, hangs, []

    def poll(self):
        return None if self.hangs and not self.stopped else self.code

    def wait(self, timeout=None):
        if self.poll() is None:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return self.code

    def terminate(self):
        self.stopped.append("terminate")

    def kill(self):
        self.stopped.append("kill")


def recording(tmp_path, video: bytes | None = VIDEO, **proc) -> clips.Recording:
    folder = tmp_path / "burst"
    folder.mkdir()
    if video is not None:
        (folder / clips.CLIP_NAME).write_bytes(video)
    return clips.Recording(FakeFfmpeg(**proc), folder, deadline=time.monotonic() + 30)


def test_a_finished_recording_is_filed_with_its_visit_and_the_temporary_folder_goes(tmp_path):
    rec = recording(tmp_path)
    src = rec.finish()
    assert src == rec.tmp / clips.CLIP_NAME and rec.problem is None
    rel = clips.keep(src, "2026/11/10/101500_ab12cd.jpg")
    assert rel == "2026/11/10/101500_ab12cd.mp4" and (config.CLIPS_DIR / rel).read_bytes() == VIDEO
    assert not list((config.CLIPS_DIR / rel).parent.glob("*.tmp"))
    rec.discard()
    rec.discard()  # twice is fine
    assert not rec.tmp.exists()


@pytest.mark.parametrize("video, proc, problem", [
    (VIDEO, {"code": 1}, "ffmpeg exited with 1"),   # the stream broke: an .mp4 without its index won't play
    (b"tiny", {}, "the recording is empty"),
    (None, {}, "the recording is empty"),
])
def test_a_failed_recording_gives_no_clip_and_says_why(tmp_path, video, proc, problem):
    rec = recording(tmp_path, video, **proc)
    assert rec.finish() is None and rec.problem == problem


def test_a_recording_past_its_deadline_is_stopped(tmp_path):
    rec = recording(tmp_path, hangs=True)
    rec.deadline = time.monotonic() - 1
    assert rec.finish() is None and rec.problem == "the recording did not finish in time"
    assert rec.proc.stopped == ["terminate"]  # asked first, so ffmpeg closes the camera's stream


def test_discarding_stops_an_ffmpeg_that_is_still_recording(tmp_path):
    rec = recording(tmp_path, hangs=True)
    rec.discard()
    assert rec.proc.stopped == ["terminate"] and not rec.tmp.exists()


def clip_visit(add_visit, conn, n: int, days_old: float, **fields) -> int:
    """A visit `days_old` days before FROZEN_NOW with a clip file on disk."""
    at = FROZEN_NOW - timedelta(days=days_old)
    image = f"{at:%Y/%m/%d}/{at:%H%M%S}_{n:06d}.jpg"
    vid = add_visit(captured_at=at.isoformat(), image_path=image, thumb_path=image.replace(".jpg", "_t.jpg"), **fields)
    rel = clips.rel_for(image)
    (config.CLIPS_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
    (config.CLIPS_DIR / rel).write_bytes(VIDEO)
    db.set_clip(conn, vid, rel)
    return vid


def with_clips(conn) -> set[int]:
    rows = conn.execute("SELECT id, clip_path FROM visits WHERE clip_path IS NOT NULL").fetchall()
    assert all((config.CLIPS_DIR / r["clip_path"]).is_file() for r in rows)
    return {r["id"] for r in rows}


def test_prune_deletes_old_clips_but_never_the_visit(conn, add_visit):
    old = clip_visit(add_visit, conn, 1, 30)
    new = clip_visit(add_visit, conn, 2, 3)
    old_file = config.CLIPS_DIR / db.get_visit(conn, old)["clip_path"]
    r = clips.prune(conn, FROZEN_NOW, days=21, per_species=0)
    assert (r["deleted"], r["orphans"], r["freed_bytes"]) == (1, 0, len(VIDEO))
    assert with_clips(conn) == {new} and not old_file.exists()
    assert db.get_visit(conn, old)["clip_path"] is None and db.count_visits(conn) == 2


def test_prune_keeps_starred_visits_clips_and_each_species_newest_ten(conn, add_visit):
    juncos = [clip_visit(add_visit, conn, i, 30 + i) for i in range(12)]   # twelve old junco clips, 0 the newest
    starred = juncos[11]                                                   # the oldest of all is a favorite
    db.set_keep(conn, starred, True)
    thrush = clip_visit(add_visit, conn, 50, 200, species="Varied Thrush")  # one clip, long ago
    r = clips.prune(conn, FROZEN_NOW, days=21, per_species=10)
    assert r["deleted"] == 1
    assert with_clips(conn) == set(juncos[:10]) | {starred, thrush}


def test_a_corrected_species_is_the_one_whose_ten_it_counts_toward(conn, add_visit):
    # Claude said junco, the owner said Bushtit: it is the only Bushtit clip, so it stays.
    fixed = clip_visit(add_visit, conn, 1, 60, corrected_species="Bushtit", corrected_visitor="bird")
    others = [clip_visit(add_visit, conn, 10 + i, 40 + i) for i in range(2)]
    clips.prune(conn, FROZEN_NOW, days=21, per_species=1)
    assert with_clips(conn) == {fixed, others[0]}


def test_a_visit_corrected_to_no_animal_is_no_species_clip(conn, add_visit):
    # Old clips of visits that turned out empty, or that nobody could name, go by age: they are nobody's ten.
    empty = clip_visit(add_visit, conn, 1, 40, corrected_visitor="none")
    other = clip_visit(add_visit, conn, 2, 40, visitor="other", species=None)
    junco = clip_visit(add_visit, conn, 3, 40)
    clips.prune(conn, FROZEN_NOW, days=21, per_species=10)
    assert with_clips(conn) == {junco}
    assert db.get_visit(conn, empty)["clip_path"] is None and db.get_visit(conn, other)["clip_path"] is None


def test_a_dry_run_changes_nothing(conn, add_visit):
    old = clip_visit(add_visit, conn, 1, 30)
    r = clips.prune(conn, FROZEN_NOW, days=21, per_species=0, dry_run=True)
    assert r["deleted"] == 1 and r["freed_bytes"] == len(VIDEO) and with_clips(conn) == {old}


def test_zero_days_keeps_every_clip(conn, add_visit):
    old = clip_visit(add_visit, conn, 1, 400)
    assert clips.prune(conn, FROZEN_NOW, days=0, per_species=0)["deleted"] == 0 and with_clips(conn) == {old}


def test_a_clip_file_no_visit_names_is_removed_once_it_is_an_hour_old(conn, add_visit):
    kept = clip_visit(add_visit, conn, 1, 1)
    stray, fresh = config.CLIPS_DIR / "2026/01/01/gone.mp4", config.CLIPS_DIR / "2026/01/01/just_now.mp4"
    stray.parent.mkdir(parents=True)
    for path, age in ((stray, 2 * clips.ORPHAN_SECONDS), (fresh, 10)):
        path.write_bytes(VIDEO)
        os.utime(path, (FROZEN_NOW.timestamp() - age,) * 2)  # fresh: a clip being filed, its visit not yet updated
    r = clips.prune(conn, FROZEN_NOW, days=21, per_species=0)
    assert (r["deleted"], r["orphans"]) == (0, 1)
    assert not stray.exists() and fresh.exists() and with_clips(conn) == {kept}


def test_usage_counts_the_files_and_the_starred(conn, add_visit):
    a = clip_visit(add_visit, conn, 1, 1)
    clip_visit(add_visit, conn, 2, 2)
    db.set_keep(conn, a, True)
    u = clips.usage(conn)
    assert (u["count"], u["bytes"], u["starred"]) == (2, 2 * len(VIDEO), 1)
    assert clips.usage(conn, config.CLIPS_DIR / "nowhere")["count"] == 0


def test_delete_ignores_missing_files_and_paths_outside_the_folder(conn, add_visit):
    vid = clip_visit(add_visit, conn, 1, 1)
    rel = db.get_visit(conn, vid)["clip_path"]
    outside = config.DATA_DIR / "keep.mp4"
    outside.write_bytes(VIDEO)
    assert clips.delete(rel) == len(VIDEO) and clips.delete(rel) == 0 and clips.delete(None) == 0
    assert clips.delete("../keep.mp4") == 0 and outside.exists()


def test_the_clip_filter_lists_only_visits_with_a_clip(conn, add_visit):
    has = clip_visit(add_visit, conn, 1, 1)
    add_visit()
    assert [v["id"] for v in db.list_visits(conn, kind="clip")] == [has]
    assert db.count_visits(conn, kind="clip") == 1 and db.count_visits(conn) == 2


def test_the_daily_run_prunes_clips_and_survives_a_failure(conn, add_visit, monkeypatch):
    old = clip_visit(add_visit, conn, 1, config.CLIP_DAYS + 5)
    newer = [clip_visit(add_visit, conn, 10 + i, 1) for i in range(config.CLIP_KEEP_PER_SPECIES)]
    now = datetime.combine(FROZEN_NOW.date(), datetime.min.time()).replace(hour=retention.DAILY_HOUR + 1)
    now = max(now, FROZEN_NOW)
    assert retention.tick(conn, now) == "daily"
    assert with_clips(conn) == set(newer) and db.get_visit(conn, old)["clip_path"] is None

    def broken(*args, **kwargs):
        raise OSError("disk went away")
    monkeypatch.setattr(clips, "prune", broken)
    retention.prune_clips(conn, now)  # logged, never raised: the photo budget's own work goes on
