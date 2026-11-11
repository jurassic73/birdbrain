"""Short video clips of real visits.

The burst's own ffmpeg also saves the next few seconds of the main stream exactly as the camera sends it (a stream
copy: no re-encoding, so it costs the Pi nothing and the camera one connection, as before). Video only, never
sound: the microphone would pick up the neighbours. The watcher keeps the clip only when the visit turns out to be
real; a false trigger's recording is thrown away.

Clips live under config.CLIPS_DIR in the same day folders as the photos (<photo stem>.mp4), outside the photo budget.
visits.clip_path names a visit's clip. The public mirror gets a clip only once the person check has looked through
it and seen nobody (birdbrain/privacy.py, birdbrain/mirror.py); the others play on the Pi's own site only. Pruning (once a day, from the photo
budget's thread) deletes clips older than CLIP_DAYS, except: a starred visit's (a favorite or a species cover,
visits.keep) and the newest CLIP_KEEP_PER_SPECIES of each species, however old. Starred clips are kept on top of
those: they take none of a species' places.

CLI: python -m birdbrain.clips [--prune] [--dry-run]    how many clips there are, and what pruning would delete
"""

import argparse
import logging
import os
import shutil
import sqlite3
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path

from . import config, db

log = logging.getLogger("birdbrain.clips")

CLIP_NAME = "clip.mp4"   # the recording's name in the burst's temporary folder
SUFFIX = ".mp4"
PARTIAL = SUFFIX + ".tmp"  # a clip while keep() copies it in; one left behind (a power cut) is swept by prune()
MIN_BYTES = 20_000       # anything smaller is a header with no picture worth keeping
STOP_SECONDS = 3         # how long ffmpeg gets to close the stream when asked to stop, before it is killed
ORPHAN_SECONDS = 3600    # a clip file no visit names is deleted once it is this old (its visit was deleted)


class Recording:
    """The burst's ffmpeg, still recording the clip after the frames are in. finish() waits for the clip;
    discard() stops whatever is still running and removes the temporary folder, and is safe to call twice."""

    def __init__(self, proc: subprocess.Popen, tmp: Path, deadline: float) -> None:
        self.proc, self.tmp, self.deadline = proc, Path(tmp), deadline
        self.problem: str | None = None  # why finish() gave no clip, for the log

    def finish(self) -> Path | None:
        """The finished clip's path in the temporary folder, or None (see .problem) when the recording failed,
        ran past its deadline or is too small to be a video."""
        try:
            code = self.proc.wait(timeout=max(0.0, self.deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.problem = "the recording did not finish in time"
            self.stop()
            return None
        path = self.tmp / CLIP_NAME
        if code != 0:
            self.problem = f"ffmpeg exited with {code}"
        elif not path.is_file() or path.stat().st_size < MIN_BYTES:
            self.problem = "the recording is empty"
        else:
            return path
        return None

    def stop(self) -> None:
        """End ffmpeg if it is still running: asked first, so it closes the camera's stream, then killed."""
        if self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=STOP_SECONDS)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=STOP_SECONDS)

    def discard(self) -> None:
        self.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


def rel_for(image_rel: str) -> str:
    """Where a visit's clip goes under the clips folder: beside the photo's name, as .mp4."""
    return image_rel.removesuffix(".jpg") + SUFFIX


def keep(src: Path, image_rel: str, clips_dir: Path | None = None) -> str:
    """Move a finished recording into the clips folder for the visit with this photo. Returns its relative path.
    Copied under a temporary name and renamed, so the site never serves half a clip."""
    clips_dir = clips_dir or config.CLIPS_DIR
    rel = rel_for(image_rel)
    dst = clips_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + PARTIAL)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return rel


def inside(rel: str) -> bool:
    """Whether a stored clip path stays under the clips folder: relative, no '..' step."""
    parts = Path(rel).parts
    return bool(parts) and not Path(rel).is_absolute() and ".." not in parts


def delete(clip_rel: str | None, clips_dir: Path | None = None) -> int:
    """Delete one clip file (missing is fine). Returns the bytes freed."""
    if not clip_rel or not inside(clip_rel):
        return 0
    path = (clips_dir or config.CLIPS_DIR) / clip_rel
    try:
        size = path.stat().st_size
        path.unlink()
    except OSError:
        return 0
    return size


def prunable(rows: Iterable[tuple], cutoff: str, per_species: int) -> set[int]:
    """Which visits lose their clip. `rows` are (visit id, species label, captured_at, starred), one per animal
    in each visit that has a clip (a visit with two kinds of animal has two rows). A clip stays when its visit is
    starred, when it is newer than `cutoff` (an ISO time), or when it is one of the newest `per_species` clips of
    any species in it, however old: every species keeps a few clips to look back on. A starred visit's clip is
    kept anyway, so it takes none of those places: the newest `per_species` are counted among the others."""
    rows = list(rows)
    safe: set[int] = set()
    by_label: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for vid, label, at, starred in rows:
        if starred or at >= cutoff:
            safe.add(vid)
        if label and not starred:
            by_label[label].add((at, vid))
    if per_species > 0:
        for clips in by_label.values():
            safe.update(vid for _, vid in sorted(clips, reverse=True)[:per_species])
    return {vid for vid, *_ in rows} - safe


def _rows(conn: sqlite3.Connection) -> list[tuple]:
    """prunable()'s rows: every visit that still has a clip, once with no label (so it goes by age when nothing
    else holds it), plus a row for every species in it, as the stats count them (labels use the owner's
    correction; a visit corrected to no animal, or an "other", counts toward no species' clips)."""
    visits = conn.execute("SELECT id, captured_at, keep IS 1 FROM visits WHERE clip_path IS NOT NULL").fetchall()
    starred = {vid for vid, _, kept in visits if kept}
    seen = conn.execute(f"SELECT id, label, captured_at FROM ({db._seen(f'clip_path IS NOT NULL AND {db.REAL}')}) "
                        f"WHERE {db.SPECIES_ONLY}").fetchall()
    return ([(vid, None, at, vid in starred) for vid, at, _ in visits]
            + [(r["id"], r["label"], r["captured_at"], r["id"] in starred) for r in seen])


def _files(clips_dir: Path, suffixes: tuple[str, ...] = (SUFFIX,)) -> list[tuple[str, int, float]]:
    """(relative path, bytes, modified time) of every clip file on disk."""
    out = []
    for root, _, names in os.walk(clips_dir):
        for name in names:
            if not name.endswith(suffixes):
                continue
            path = Path(root) / name
            try:
                st = path.stat()
            except OSError:
                continue
            out.append((str(path.relative_to(clips_dir)), st.st_size, st.st_mtime))
    return out


def prune(conn: sqlite3.Connection, now: datetime | None = None, *, days: int | None = None,
          per_species: int | None = None, clips_dir: Path | None = None, dry_run: bool = False) -> dict:
    """Delete the clips past their time (see prunable) and the clip files no visit names any more. The visits
    stay; only visits.clip_path is cleared. Returns {"deleted", "orphans", "freed_bytes"}; with dry_run nothing
    changes and the numbers say what would go. days 0 keeps every clip (orphans are still removed)."""
    now = now or datetime.now()
    days = config.CLIP_DAYS if days is None else days
    per_species = config.CLIP_KEEP_PER_SPECIES if per_species is None else per_species
    clips_dir = clips_dir or config.CLIPS_DIR
    deleted = orphans = freed = 0
    if days > 0:
        cutoff = (now - timedelta(days=days)).isoformat(timespec="seconds")
        for vid in sorted(prunable(_rows(conn), cutoff, per_species)):
            row = conn.execute("SELECT clip_path FROM visits WHERE id = ?", (vid,)).fetchone()
            if not row or not row[0]:
                continue
            if not dry_run:
                # Checked again under the write lock: a visit starred since the list was made keeps its clip.
                with db.writing(conn):
                    cur = conn.execute("UPDATE visits SET clip_path = NULL WHERE id = ? AND keep IS NOT 1", (vid,))
                if cur.rowcount != 1:
                    continue
                freed += delete(row[0], clips_dir)
            else:
                try:
                    freed += (clips_dir / row[0]).stat().st_size if inside(row[0]) else 0
                except OSError:
                    pass
            deleted += 1
    named = {r[0] for r in conn.execute("SELECT clip_path FROM visits WHERE clip_path IS NOT NULL")}
    for rel, size, mtime in _files(clips_dir, (SUFFIX, PARTIAL)):
        if rel not in named and now.timestamp() - mtime > ORPHAN_SECONDS:
            if not dry_run:
                delete(rel, clips_dir)
            orphans += 1
            freed += size
    return {"deleted": deleted, "orphans": orphans, "freed_bytes": freed}


def usage(conn: sqlite3.Connection, clips_dir: Path | None = None) -> dict:
    """What the clips take, for Settings > Storage: {"count", "bytes", "starred", "days", "per_species", "seconds"}."""
    files = _files(clips_dir or config.CLIPS_DIR)
    starred = conn.execute("SELECT COUNT(*) FROM visits WHERE clip_path IS NOT NULL AND keep IS 1").fetchone()[0]
    return {"count": len(files), "bytes": sum(size for _, size, _ in files), "starred": starred,
            "days": config.CLIP_DAYS, "per_species": config.CLIP_KEEP_PER_SPECIES, "seconds": config.CLIP_SECONDS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visit clips: how many there are, and pruning the old ones.")
    parser.add_argument("--prune", action="store_true", help="delete clips past their time now")
    parser.add_argument("--dry-run", action="store_true", help="with --prune: only say what would be deleted")
    args = parser.parse_args()
    with db.session() as conn:
        if args.prune:
            r = prune(conn, dry_run=args.dry_run)
            print(f"{'would delete' if args.dry_run else 'deleted'} {r['deleted']} clips past {config.CLIP_DAYS} days "
                  f"and {r['orphans']} with no visit, {r['freed_bytes'] / 1e6:.1f} MB")
        u = usage(conn)
        print(f"{u['count']} clips, {u['bytes'] / 1e6:.1f} MB ({u['starred']} of starred visits)")


if __name__ == "__main__":
    main()
