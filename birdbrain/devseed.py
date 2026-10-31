"""Fill the local database with fake visits for UI work: python -m birdbrain.devseed [--reset]"""

import argparse
import io
import json
import random
import shutil
from datetime import datetime, timedelta

from PIL import Image, ImageOps

from . import config, db, images

BIRDS = [
    ("Dark-eyed Junco", "Oregon race"),
    ("Black-capped Chickadee", ""),
    ("American Robin", ""),
    ("Steller's Jay", "crest raised"),
    ("Bushtit", "small flock"),
    ("Lesser Goldfinch", ""),
    ("American Goldfinch", "winter plumage"),
    ("Song Sparrow", ""),
    ("Spotted Towhee", ""),
    ("Anna's Hummingbird", "male, iridescent gorget"),
    ("House Finch", ""),
    ("Northern Flicker", "red-shafted"),
    ("California Scrub-Jay", ""),
]
BIRD_WEIGHTS = [14, 12, 9, 6, 5, 5, 4, 6, 5, 4, 6, 2, 3]


def frame_bytes(src: Image.Image, night: bool) -> bytes:
    img = ImageOps.grayscale(src).convert("RGB") if night else src
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete existing dev data first")
    parser.add_argument("--count", type=int, default=90)
    args = parser.parse_args()

    if args.reset:
        if config.ROOT not in config.DATA_DIR.resolve().parents:
            raise SystemExit(f"refusing to reset data outside the repo: {config.DATA_DIR}")
        for suffix in ("", "-wal", "-shm"):
            config.DB_PATH.with_name(config.DB_PATH.name + suffix).unlink(missing_ok=True)
        shutil.rmtree(config.IMAGES_DIR, ignore_errors=True)

    sources = sorted((config.ROOT / "snapshots").glob("*ch0*.jpg"))
    if not sources:
        raise SystemExit("no snapshots/*ch0*.jpg to seed from; run scripts/snap.sh first")
    frames = [Image.open(p).convert("RGB") for p in sources]

    conn = db.connect()
    db.init_db(conn)
    rng = random.Random(42)
    now = datetime.now().replace(microsecond=0)

    for _ in range(args.count):
        captured = now - timedelta(days=rng.uniform(0, 40))
        night = captured.hour >= 21 or captured.hour < 6
        roll = rng.random()
        if roll < 0.15:
            visitor, species, notes, count = "none", None, "", None
        elif night or roll < 0.25:
            visitor = rng.choice(["raccoon", "possum", "squirrel"] if night else ["squirrel"])
            species, notes, count = None, "", rng.choice([1, 1, 2])
        else:
            visitor = "bird"
            species, notes = rng.choices(BIRDS, BIRD_WEIGHTS)[0]
            count = rng.choice([1, 1, 1, 2, 3])
        confidence = rng.choices(["high", "medium", "low"], [70, 20, 10])[0] if visitor != "none" else None
        escalated = confidence == "low" and rng.random() < 0.6
        result = {"visitor": visitor, "species": species, "confidence": confidence, "count": count, "notes": notes}
        image_rel, thumb_rel = images.store_frame(frame_bytes(rng.choice(frames), night), captured)
        db.insert_visit(
            conn,
            captured_at=captured.isoformat(),
            image_path=image_rel,
            thumb_path=thumb_rel,
            visitor=visitor,
            species=species,
            count=count,
            confidence=confidence,
            notes=notes or None,
            model="claude-sonnet-5-5" if escalated else "claude-haiku-5-5",
            escalated=int(escalated),
            needs_review=int(confidence == "low"),
            is_night=int(night),
            raw_json=json.dumps(result),
        )
    print(f"seeded {args.count} visits into {config.DB_PATH}")


if __name__ == "__main__":
    main()
