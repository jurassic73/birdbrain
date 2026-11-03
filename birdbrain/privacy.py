"""Keep photos of people off the public mirror (scripts/push_public.sh). The Pi's own site is never filtered.

The YOLOX-tiny COCO detector (detector.py) looks for a "person" on the whole frame and on four overlapping
quarters, so someone small on the far lawn still registers. The bar is low on purpose: a false "person"
only keeps one photo off the mirror. Everything fails closed: a photo that can't be checked isn't pushed.

Scores are cached per photo in DATA_DIR/privacy.json, so each file is checked once.

"Make public": when the owner looks at a held-back visit and sees no person, the visit page pins the
scores its photos had then in visits.public_ok ({"at": time, "files": {photo: score}}). Only those files at
no more than those scores are let through; a new file, or one that scores higher when checked again, holds
the visit back again. An override that can't be read counts as none (fails closed).
CLI (exit 0 = clean, 1 = person or not checked, 2 = error):
  python -m birdbrain.privacy check live.jpg          is it safe to push this frame?
  python -m birdbrain.privacy filter mirror.db excl   drop visits with a person from a copy of the database
                                                       and list their files in excl (birdbrain.mirror runs this)
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from . import config, images

MIN_SCORE = float(os.environ.get("PRIVACY_MIN_SCORE", "0.25"))
FILTER_BUDGET = float(os.environ.get("PRIVACY_BUDGET", "90"))  # seconds of checking per push; the rest waits
_det = None


def _detector():
    global _det
    if _det is None:
        from .detector import Detector

        if not config.DETECTOR_MODEL.exists():
            raise FileNotFoundError(f"no detector model at {config.DETECTOR_MODEL}")
        _det = Detector(config.DETECTOR_MODEL)
    return _det


def views(bgr):
    """The whole frame plus four quarters overlapping by a tenth, each seen by the detector on its own."""
    h, w = bgr.shape[:2]
    hh, hw, oy, ox = h // 2, w // 2, h // 10, w // 10
    quarters = [bgr[y0:y0 + hh + oy, x0:x0 + hw + ox] for y0 in (0, hh - oy) for x0 in (0, hw - ox)]
    return [bgr, *quarters]


def person_score(bgr) -> float:
    """Highest "person" score anywhere in the frame (0 when none)."""
    det = _detector()
    best = 0.0
    for view in views(bgr):
        for d in det.detect(view, min(MIN_SCORE, config.DETECTOR_MIN_SCORE)):
            if d.label == "person":
                best = max(best, d.score)
    return round(best, 3)


def score_file(path: Path) -> float:
    import cv2  # here, so the web app (which only reads the cache) never loads OpenCV

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"can't read {path}")
    return person_score(img)


def cache_path() -> Path:
    return config.DATA_DIR / "privacy.json"


def load_cache() -> dict[str, float]:
    try:
        return json.loads(cache_path().read_text())
    except (FileNotFoundError, ValueError):
        return {}


def save_cache(cache: dict[str, float]) -> None:
    """Merge with what's on disk, so two runs at once (cron and a backfill) never drop each other's scores."""
    cache = {**load_cache(), **cache}
    tmp = cache_path().with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cache, separators=(",", ":")))
    tmp.replace(cache_path())


def parse_override(text: str | None) -> dict[str, float]:
    """The photo scores the owner cleared with Make public ({} when none, or when it can't be read)."""
    try:
        files = json.loads(text)["files"] if text else {}
        return {str(rel): float(score) for rel, score in files.items()}
    except Exception:
        return {}


def held_back(rel: str, score: float, override: dict[str, float]) -> bool:
    """A person score that keeps the visit off the mirror: over the bar, and not cleared by the owner for
    this very file at this score or higher."""
    return score >= MIN_SCORE and not (rel in override and score <= override[rel])


def make_override(image_rel: str, cache: dict[str, float], thumb_rel: str | None = None) -> dict | None:
    """What Make public stores: the scores of the visit's photos that showed a person. None unless every
    photo has been checked and at least one held the visit back (there's nothing to clear otherwise)."""
    files = {}
    for rel in files_to_check(image_rel, thumb_rel):
        if rel not in cache:
            return None
        if cache[rel] >= MIN_SCORE:
            files[rel] = cache[rel]
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": files} if files else None


_status: tuple[float, dict] = (0.0, {})


def visit_status(image_rel: str, public_ok: str | None = None, thumb_rel: str | None = None) -> str | None:
    """'person', 'unchecked', 'cleared' (the owner said Make public) or None (clean) for a visit's
    photos, for the pill on the Pi's visit page."""
    global _status
    try:
        mtime = cache_path().stat().st_mtime
    except FileNotFoundError:
        return None  # the mirror filter has never run: nothing to report
    if mtime != _status[0]:
        _status = (mtime, load_cache())
    cache, override = _status[1], parse_override(public_ok)
    scores = {rel: cache.get(rel) for rel in files_to_check(image_rel, thumb_rel)}
    if any(s is not None and held_back(rel, s, override) for rel, s in scores.items()):
        return "person"
    if None in scores.values():
        return "unchecked"
    return "cleared" if any(s >= MIN_SCORE for s in scores.values()) else None


def visit_files(image_rel: str, thumb_rel: str | None = None, thumbs: bool = True) -> list[str]:
    """A visit's photo and burst frames (and its thumbnail, which needs no check of its own)."""
    files = [image_rel, *images.burst_paths(image_rel)]
    return files + [thumb_rel] if thumbs and thumb_rel else files


def files_to_check(image_rel: str, thumb_rel: str | None) -> list[str]:
    """The files of a visit that could reach the mirror, so each needs a score: the photo and its burst frames,
    or, once the photo budget has left only the thumbnail (retention.py), the thumbnail in the photo's place.
    A file that isn't on disk can't leak anything and isn't listed."""
    files = visit_files(image_rel, thumbs=False)
    if not (config.IMAGES_DIR / image_rel).exists() and thumb_rel:
        files = [thumb_rel, *files[1:]]
    return [rel for rel in files if (config.IMAGES_DIR / rel).exists()]


def blocked_visits(rows, cache: dict[str, float], budget: float = FILTER_BUDGET) -> tuple[list, int]:
    """Check unchecked photos (newest visits first, within the time budget) and return the visits to keep
    off the mirror: a person in any frame (unless the owner cleared it), or not checked yet, which includes a
    file on disk that can't be read. Rows are (id, image, thumb[, public_ok]). Also returns how many photos
    were checked."""
    start, checked = time.monotonic(), 0
    blocked = []
    for vid, image_rel, thumb_rel, *rest in rows:
        override = parse_override(rest[0] if rest else None)
        person = unchecked = False
        for rel in files_to_check(image_rel, thumb_rel):
            if rel not in cache:
                if time.monotonic() - start > budget:
                    unchecked = True
                    continue
                try:
                    cache[rel] = score_file(config.IMAGES_DIR / rel)
                except ValueError:  # a file that isn't a readable image: unchecked, so the visit stays off
                    unchecked = True
                    continue
                checked += 1
            person |= held_back(rel, cache[rel], override)
        if person or unchecked:
            blocked.append(vid)
    return blocked, checked


def filter_db(db_path: Path, exclude_path: Path) -> int:
    """Remove blocked visits from a copy of the database; list their files for rsync. Returns how many."""
    cache = load_cache()
    db = sqlite3.connect(db_path)
    try:
        rows = db.execute("SELECT id, image_path, thumb_path, public_ok FROM visits ORDER BY id DESC").fetchall()
    except sqlite3.OperationalError:  # a database from before Make public: no overrides
        rows = db.execute("SELECT id, image_path, thumb_path FROM visits ORDER BY id DESC").fetchall()
    try:
        blocked, checked = blocked_visits(rows, cache)
    finally:
        save_cache(cache)  # keep what was checked even if a later photo failed
    by_id = {r[0]: r for r in rows}
    lines = []
    for vid in blocked:
        _, image_rel, thumb_rel, *_ = by_id[vid]
        lines += ["/" + rel for rel in visit_files(image_rel, thumb_rel)]
    cleared = sum(1 for r in rows if len(r) > 3 and r[0] not in blocked and parse_override(r[3]))
    db.executemany("DELETE FROM visits WHERE id = ?", [(v,) for v in blocked])
    try:
        db.execute("UPDATE visits SET public_ok = NULL")  # the owner's privacy calls stay on the Pi
    except sqlite3.OperationalError:
        pass
    for table in ("embeddings",):  # rows keyed by visit that would otherwise outlive it
        try:
            db.execute(f"DELETE FROM {table} WHERE visit_id NOT IN (SELECT id FROM visits)")
        except sqlite3.OperationalError:
            pass
    db.commit()
    db.execute("VACUUM")
    db.close()
    exclude_path.write_text("".join(f"{line}\n" for line in lines))
    print(f"privacy: {checked} photos checked, {len(blocked)} visits kept off the mirror, "
          f"{cleared} let through by Make public")
    return len(blocked)


def main(argv: list[str]) -> int:
    try:
        if len(argv) == 2 and argv[0] == "check":
            score = score_file(Path(argv[1]))
            print(f"person {score}" if score >= MIN_SCORE else f"clean {score}")
            return 1 if score >= MIN_SCORE else 0
        if len(argv) == 3 and argv[0] == "filter":
            filter_db(Path(argv[1]), Path(argv[2]))
            return 0
    except Exception as e:  # fail closed: the push script stops on a nonzero exit
        print(f"privacy check failed: {e}", file=sys.stderr)
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
