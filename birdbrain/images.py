import argparse
import io
import os
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageStat

from . import config

# Long edge of a thumbnail. A grid tile is ~233x175 CSS px on the owner's laptop (~470x350 device px at 2x) and
# fills it with object-fit: cover, so a 16:9 frame needs ~620x350; 800 also covers a phone's 2-column grid at 3x.
# At 400 the browser stretched every tile ~1.5x, which is what made them soft (2026-11-02).
THUMB_PX = 800
THUMB_QUALITY = 88
# Light, after the Lanczos shrink: makes up for the browser's own resize to the tile without halos.
THUMB_SHARPEN = ImageFilter.UnsharpMask(radius=0.6, percent=50, threshold=2)
DOT = "\u00b7"


def _stamp_font(size: int):
    for path in config.STAMP_FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def stamp_text(when: datetime) -> str:
    """'Sun Nov 1  ·  9:33 AM'"""
    return f"{when:%a %b} {when.day}  {DOT}  {when.hour % 12 or 12}:{when:%M %p}"


def _pill(size: tuple[int, int], when: datetime):
    """The stamp's scale, font, text and pill box (x, y, right, bottom) on a photo of this size."""
    s = size[1] / 1080
    font = _stamp_font(max(10, round(34 * s)))
    text = stamp_text(when)
    h, margin = 64 * s, 36 * s
    x, y = margin, size[1] - margin - h
    return s, font, text, (x, y, x + font.getlength(text) + 84 * s, y + h)


def stamp(img: Image.Image, when: datetime) -> Image.Image:
    """Draw the capture time in the lower-left corner: a dark glass pill with a small green dot.

    Sized to the photo (a 64 px pill with 34 px text on a 1080 px frame), so smaller false-trigger photos get a smaller one.
    Only stored photos get it; Claude and the Pi's checks see the clean frame.
    """
    s, font, text, box = _pill(img.size, when)
    x, y, h = box[0], box[1], box[3] - box[1]
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle(box, radius=h / 2, fill=(15, 20, 18, 150),
                        outline=(255, 255, 255, 60), width=max(1, round(2 * s)))
    dot = 7 * s
    d.ellipse((x + 33 * s - dot, y + h / 2 - dot, x + 33 * s + dot, y + h / 2 + dot), fill=(126, 200, 140, 255))
    # Centre the ink of a capital (the digits share its height) on the pill's middle, measured on the font itself:
    # ascender/descender metrics differ by font and leave room for tails ("Sep", "Aug") the stamp rarely has.
    # getbbox and text both use the top-left anchor, so this also works for Pillow's bitmap fallback font.
    # Rounded to a whole pixel so the tops of the letters stay crisp.
    top, base = font.getbbox("H")[1::2]
    d.text((x + 56 * s, round(y + h / 2 - (top + base) / 2)), text, font=font, fill=(245, 245, 240, 255))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def store_frame(
    jpeg: bytes,
    captured_at: datetime,
    images_dir: Path | None = None,
    max_edge: int | None = None,
    quality: int = 85,
) -> tuple[str, str]:
    """Save a frame (re-encoded: the camera's near-lossless JPEGs are ~1 MB) and a thumbnail.

    Returns both paths relative to the images dir. max_edge shrinks the saved frame, e.g. for false triggers.
    """
    images_dir = images_dir or config.IMAGES_DIR
    stem = f"{captured_at:%Y/%m/%d}/{captured_at:%H%M%S}_{uuid.uuid4().hex[:6]}"
    image_rel, thumb_rel = f"{stem}.jpg", f"{stem}_t.jpg"
    (images_dir / image_rel).parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    full = img.copy()
    if max_edge:
        full.thumbnail((max_edge, max_edge))
    if config.PHOTO_STAMP:
        full = stamp(full, captured_at)
    full.save(images_dir / image_rel, "JPEG", quality=quality, optimize=True)
    make_thumb(img).save(images_dir / thumb_rel, "JPEG", quality=THUMB_QUALITY, optimize=True)
    return image_rel, thumb_rel


def make_thumb(img: Image.Image) -> Image.Image:
    """A sharp thumbnail of a clean (unstamped) frame: Lanczos down to THUMB_PX, then a light unsharp mask."""
    thumb = img.copy()
    thumb.thumbnail((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS, reducing_gap=3.0)
    return thumb.filter(THUMB_SHARPEN)


def store_burst(
    frames: list[bytes],
    image_rel: str,
    images_dir: Path | None = None,
    max_edge: int | None = None,
    quality: int = 85,
    captured_at: datetime | None = None,
) -> list[str]:
    """Save a whole burst next to its visit's photo as <stem>_b1.jpg, _b2.jpg... in capture order (no thumbnails).

    Kept for false triggers, so a doubtful one can be checked frame by frame. With captured_at, each frame gets the
    same time stamp as the photo. Returns the relative paths.
    """
    images_dir = images_dir or config.IMAGES_DIR
    stem = image_rel.removesuffix(".jpg")
    paths = []
    for i, jpeg in enumerate(frames, 1):
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if max_edge:
            img.thumbnail((max_edge, max_edge))
        if captured_at and config.PHOTO_STAMP:
            img = stamp(img, captured_at)
        rel = f"{stem}_b{i}.jpg"
        img.save(images_dir / rel, "JPEG", quality=quality, optimize=True)
        paths.append(rel)
    return paths


def burst_file(name: str) -> tuple[str, int, bool] | None:
    """What a file in a day folder is, when it belongs to a burst: (photo stem, frame number, is it the frame's
    thumbnail). <stem>_b3.jpg is frame 3; <stem>_b3_t.jpg is its thumbnail (made when the owner picks that frame as
    the visit's photo, pick_thumb). Anything else is None."""
    if not name.endswith(".jpg"):
        return None
    base = name.removesuffix(".jpg")
    thumb = base.endswith("_t")
    stem, sep, n = base.removesuffix("_t").rpartition("_b")
    if not sep or not stem or not n.isdigit():
        return None
    return stem, int(n), thumb


def burst_path(image_rel: str, n: int) -> str:
    """Where burst frame n (1-based) of a visit's photo is saved."""
    return f"{image_rel.removesuffix('.jpg')}_b{n}.jpg"


def burst_thumb_path(image_rel: str, n: int) -> str:
    """Where burst frame n's thumbnail goes once the owner picks it as the visit's photo (pick_thumb)."""
    return f"{image_rel.removesuffix('.jpg')}_b{n}_t.jpg"


def _burst_files(image_rel: str, images_dir: Path | None, thumbs: bool) -> list[str]:
    images_dir = images_dir or config.IMAGES_DIR
    stem = Path(image_rel.removesuffix(".jpg"))
    nums = sorted(parsed[1] for p in (images_dir / stem).parent.glob(f"{stem.name}_b*.jpg")
                  if (parsed := burst_file(p.name)) and parsed[0] == stem.name and parsed[2] == thumbs)
    return [(burst_thumb_path if thumbs else burst_path)(image_rel, n) for n in nums]


def burst_paths(image_rel: str, images_dir: Path | None = None) -> list[str]:
    """A visit's saved burst frames, in capture order (empty when only the best frame was kept). Never their
    thumbnails (burst_thumbs)."""
    return _burst_files(image_rel, images_dir, thumbs=False)


def burst_thumbs(image_rel: str, images_dir: Path | None = None) -> list[str]:
    """Thumbnails of a visit's burst frames on disk (at most one: the frame the owner picked as the photo)."""
    return _burst_files(image_rel, images_dir, thumbs=True)


def pick_thumb(image_rel: str, n: int, images_dir: Path | None = None) -> str:
    """Make burst frame n's thumbnail (the card's picture once the owner picks that frame as the visit's photo)
    and drop any other frame's, so a visit never carries more than one. The frames themselves are never touched.
    Returns the thumbnail's path; FileNotFoundError when frame n isn't on disk."""
    images_dir = images_dir or config.IMAGES_DIR
    rel = burst_thumb_path(image_rel, n)
    with Image.open(images_dir / burst_path(image_rel, n)) as im:
        thumb = make_thumb(im.convert("RGB"))
    tmp = (images_dir / rel).with_name(Path(rel).name + ".tmp")
    thumb.save(tmp, "JPEG", quality=THUMB_QUALITY, optimize=True)
    os.replace(tmp, images_dir / rel)
    drop_burst_thumbs(image_rel, images_dir, keep=n)
    return rel


def drop_burst_thumbs(image_rel: str, images_dir: Path | None = None, keep: int | None = None) -> None:
    """Remove a visit's burst-frame thumbnails (all but frame `keep`'s): the pick was taken back or changed."""
    images_dir = images_dir or config.IMAGES_DIR
    for rel in burst_thumbs(image_rel, images_dir):
        if keep is None or burst_file(Path(rel).name)[1] != keep:
            (images_dir / rel).unlink(missing_ok=True)


def frame_bytes(image_rel: str, thumb_rel: str, images_dir: Path | None = None) -> int:
    """Disk space used by one visit's photo, thumbnail and any burst frames (missing files count as 0)."""
    return frames_bytes([(image_rel, thumb_rel)], images_dir)[0]


def frames_bytes(visits: list[tuple[str, str]], images_dir: Path | None = None) -> list[int]:
    """frame_bytes for many (image, thumbnail) pairs at once, in the same order."""
    return [sum(parts) for parts in frame_parts(visits, images_dir)]


def frame_parts(visits: list[tuple[str, str]], images_dir: Path | None = None) -> list[tuple[int, int, int]]:
    """(photo, thumbnail, burst frames) bytes for many (image, thumbnail) pairs at once, in the same order.

    Lists each day folder once instead of globbing it per visit: with hundreds of false triggers a day,
    the per-visit glob took ~1 s on the Pi and grew with the square of the day's count.
    """
    images_dir = images_dir or config.IMAGES_DIR
    folders: dict[Path, tuple[dict[str, int], dict[str, int]]] = {}

    def folder(rel: str) -> tuple[dict[str, int], dict[str, int]]:
        parent = Path(rel).parent
        if parent not in folders:
            sizes: dict[str, int] = {}
            bursts: dict[str, int] = {}  # photo stem -> its burst frames' bytes
            try:
                with os.scandir(images_dir / parent) as entries:
                    for e in entries:
                        try:
                            size = e.stat().st_size
                        except FileNotFoundError:
                            continue
                        sizes[e.name] = size
                        if parsed := burst_file(e.name):  # a frame or its thumbnail: both count as burst
                            bursts[parsed[0]] = bursts.get(parsed[0], 0) + size
            except (FileNotFoundError, NotADirectoryError):
                pass
            folders[parent] = (sizes, bursts)
        return folders[parent]

    out = []
    for image_rel, thumb_rel in visits:
        photo = folder(image_rel)[0].get(Path(image_rel).name, 0)
        thumb = folder(thumb_rel)[0].get(Path(thumb_rel).name, 0) if thumb_rel else 0
        burst = folder(image_rel)[1].get(Path(image_rel).name.removesuffix(".jpg"), 0)
        out.append((photo, thumb, burst))
    return out


def delete_frame(image_rel: str, thumb_rel: str, images_dir: Path | None = None) -> None:
    """Delete every file of a visit: photo, thumbnail, burst frames and a picked frame's thumbnail."""
    images_dir = images_dir or config.IMAGES_DIR
    for rel in (image_rel, thumb_rel, *burst_paths(image_rel, images_dir), *burst_thumbs(image_rel, images_dir)):
        (images_dir / rel).unlink(missing_ok=True)


# Thinning an older visit's photo (birdbrain.retention): saved once, smaller, from the stored original.
THIN_QUALITY = 88


def thin_photo(image_rel: str, edge: int, images_dir: Path | None = None,
               quality: int = THIN_QUALITY) -> tuple[int, int]:
    """Save a visit's photo again with its long edge at `edge` px. Returns (bytes before, bytes after).

    Lanczos from the stored photo at quality 88 (a little above the original's 85, so the shrink isn't followed by a
    second round of visible loss), keeping any EXIF and color profile. The time stamp shrinks with the photo and stays
    readable. A photo already no bigger than `edge` is left exactly as it is, which also means a thinned photo is never
    shrunk again. Writes a temp file and renames it, so a crash never leaves half a photo. The thumbnail and any
    burst frames aren't touched.
    """
    images_dir = images_dir or config.IMAGES_DIR
    path = images_dir / image_rel
    before = path.stat().st_size
    with Image.open(path) as im:
        if max(im.size) <= edge:
            return before, before
        extra = {k: im.info[k] for k in ("exif", "icc_profile") if im.info.get(k)}
        img = im.convert("RGB")
    w, h = img.size
    scale = edge / max(w, h)
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=quality, optimize=True, **extra)
    if buf.tell() >= before:  # can't happen with a real photo, but never make one bigger
        return before, before
    tmp = path.with_name(f".{path.name}.thin.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return before, buf.tell()


def drop_photo(image_rel: str, images_dir: Path | None = None) -> None:
    """Delete a visit's photo and burst frames but keep its thumbnails (the photo's, and a picked frame's),
    so its card still shows."""
    images_dir = images_dir or config.IMAGES_DIR
    for rel in (image_rel, *burst_paths(image_rel, images_dir)):
        (images_dir / rel).unlink(missing_ok=True)


def _scaled_box(box, src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int, int, int]:
    f = dst[0] / src[0]
    return (max(0, int(box[0] * f)), max(0, int(box[1] * f)),
            min(dst[0], int(box[2] * f) + 1), min(dst[1], int(box[3] * f) + 1))


def _cover_stamp(thumb: Image.Image, full: Image.Image, old: Image.Image, when: datetime) -> bool:
    """A thumbnail remade from a stamped photo shows the stamp; old thumbnails were made from the clean frame.

    If the photo's pill corner differs clearly from the old thumbnail's (the stamp is there), paint the old thumbnail's
    clean corner over it, feathered at the edges. Returns whether it did.
    """
    s, _, _, (x, y, right, bottom) = _pill(full.size, when)
    pad = 6 * s + 2
    box = (x - pad, y - pad, right + pad + 0.1 * (right - x), bottom + pad)  # a little extra in case the font differs
    small = _scaled_box(box, full.size, old.size)
    ref = thumb.resize(old.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(ref, old)
    corner = sum(ImageStat.Stat(diff.crop(small)).mean) / 3
    rest = sum(ImageStat.Stat(diff).mean) / 3
    if corner < 2 * rest + 8:
        return False
    big = _scaled_box(box, full.size, thumb.size)
    mask = Image.new("L", thumb.size, 0)
    ImageDraw.Draw(mask).rectangle((big[0] - 3, big[1] - 3, big[2] + 3, big[3] + 3), fill=255)
    thumb.paste(old.resize(thumb.size, Image.Resampling.LANCZOS), mask=mask.filter(ImageFilter.GaussianBlur(1.5)))
    return True


def rebuild_thumb(image_rel: str, thumb_rel: str, when: datetime, images_dir: Path | None = None,
                  dry_run: bool = False, force: bool = False) -> tuple[str, int, int]:
    """Remake one thumbnail from its stored photo the current way. Returns (status, old bytes, new bytes).

    status: "rebuilt", "stamp" (rebuilt, stamp covered), "current" (already THUMB_PX-sized; force redoes it) or
    "missing" (photo or thumbnail gone; nothing is written). Writes a temp file and renames it, so the site never serves
    half a thumbnail. Never touches the photo or burst frames.
    """
    images_dir = images_dir or config.IMAGES_DIR
    full_path, thumb_path = images_dir / image_rel, images_dir / thumb_rel
    try:
        with Image.open(thumb_path) as t, Image.open(full_path) as f:
            if max(t.size) == min(THUMB_PX, max(f.size)) and not force:  # sizes are read without decoding
                return "current", thumb_path.stat().st_size, thumb_path.stat().st_size
            old, full = t.convert("RGB"), f.convert("RGB")
        old_bytes = thumb_path.stat().st_size
    except FileNotFoundError:
        return "missing", 0, 0
    thumb = make_thumb(full)
    status = "stamp" if _cover_stamp(thumb, full, old, when) else "rebuilt"
    buf = io.BytesIO()
    thumb.save(buf, "JPEG", quality=THUMB_QUALITY, optimize=True)
    if not dry_run:
        tmp = thumb_path.with_name(thumb_path.name + ".tmp")
        try:
            tmp.write_bytes(buf.getvalue())
            os.replace(tmp, thumb_path)
        finally:
            tmp.unlink(missing_ok=True)
    return status, old_bytes, buf.tell()


def _rebuild_row(job: tuple) -> tuple[str, str, int, int, str]:
    """rebuild_thumb for one row, picklable for worker processes. A failure comes back as status "failed"."""
    (image_rel, thumb_rel, captured_at), images_dir, dry_run, force = job
    try:
        return (thumb_rel, *rebuild_thumb(image_rel, thumb_rel, datetime.fromisoformat(captured_at), images_dir,
                                          dry_run, force), "")
    except Exception as e:
        return thumb_rel, "failed", 0, 0, str(e)


def rebuild_thumbs(rows: Iterable[tuple[str, str, str]], images_dir: Path | None = None, dry_run: bool = False,
                   force: bool = False, log: Callable[[str], None] = print, every: int = 200,
                   seconds: float = 30, jobs: int = 1) -> Counter:
    """rebuild_thumb for (image_path, thumb_path, captured_at) rows, logging progress every `every` files or `seconds`.

    jobs > 1 spreads the files over that many worker processes (each file is independent); jobs=1 runs in this process.
    """
    rows = list(rows)
    images_dir = images_dir or config.IMAGES_DIR  # resolved here so workers need no config of their own
    work = ((row, images_dir, dry_run, force) for row in rows)
    counts: Counter = Counter()
    start = last = time.monotonic()
    pool = ProcessPoolExecutor(jobs) if jobs > 1 else None
    try:
        results = pool.map(_rebuild_row, work, chunksize=8) if pool else map(_rebuild_row, work)
        for i, (thumb_rel, status, before, after, err) in enumerate(results, 1):
            if err:
                log(f"failed: {thumb_rel}: {err}")
            counts[status] += 1
            counts["bytes_before"] += before
            counts["bytes_after"] += after
            now = time.monotonic()
            if i % every == 0 or now - last >= seconds or i == len(rows):
                last = now
                left = (now - start) / i * (len(rows) - i)
                eta = f"~{left / 60:.0f} min left" if left >= 90 else f"~{left:.0f} s left"
                log(f"{i}/{len(rows)} done, {counts['failed']} failed, {eta}")
    finally:
        if pool:  # Ctrl-C lands here too: drop the queued files; thumbnails already written are fine (idempotent)
            pool.shutdown(wait=False, cancel_futures=True)
    return counts


def main() -> None:
    from . import db

    parser = argparse.ArgumentParser(prog="python -m birdbrain.images")
    parser.add_argument("--rebuild-thumbs", action="store_true", required=True,
                        help="remake every visit's thumbnail from its stored photo (newest first)")
    parser.add_argument("--dry-run", action="store_true", help="work out the new sizes but write nothing")
    parser.add_argument("--force", action="store_true", help="also redo thumbnails that are already THUMB_PX")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                        help="worker processes (default: CPUs - 1; 1 = one file at a time)")
    args = parser.parse_args()
    with db.session() as conn:
        rows = [tuple(r) for r in conn.execute(
            "SELECT image_path, thumb_path, captured_at FROM visits ORDER BY id DESC")]
    print(f"{len(rows)} visits in {config.IMAGES_DIR}{' (dry run: nothing is written)' if args.dry_run else ''}")
    c = rebuild_thumbs(rows, dry_run=args.dry_run, force=args.force, jobs=args.jobs)
    mb = lambda n: f"{n / 1e6:.1f} MB"
    print(f"{'would rebuild' if args.dry_run else 'rebuilt'} {c['rebuilt'] + c['stamp']} "
          f"({c['stamp']} with the time stamp covered), {c['current']} already sharp, {c['missing']} missing, "
          f"{c['failed']} failed; thumbnails {mb(c['bytes_before'])} -> {mb(c['bytes_after'])} "
          f"({mb(c['bytes_after'] - c['bytes_before'])} more)")


if __name__ == "__main__":
    main()
