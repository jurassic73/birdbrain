import io
import os
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config

THUMB_PX = 400
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


def stamp(img: Image.Image, when: datetime) -> Image.Image:
    """Draw the capture time in the lower-left corner: a dark glass pill with a small green dot.

    Sized to the photo (a 64 px pill on a 1080 px frame), so smaller false-trigger photos get a smaller one.
    Only stored photos get it; Claude and the Pi's checks see the clean frame.
    """
    s = img.height / 1080
    font = _stamp_font(max(10, round(38 * s)))
    text = stamp_text(when)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    w = d.textlength(text, font=font)
    h, margin = 64 * s, 36 * s
    x, y = margin, img.height - margin - h
    d.rounded_rectangle((x, y, x + w + 84 * s, y + h), radius=h / 2, fill=(15, 20, 18, 150),
                        outline=(255, 255, 255, 60), width=max(1, round(2 * s)))
    dot = 7 * s
    d.ellipse((x + 33 * s - dot, y + h / 2 - dot, x + 33 * s + dot, y + h / 2 + dot), fill=(126, 200, 140, 255))
    d.text((x + 56 * s, y + h / 2), text, font=font, fill=(245, 245, 240, 255), anchor="lm")
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
    img.thumbnail((THUMB_PX, THUMB_PX))
    img.save(images_dir / thumb_rel, "JPEG", quality=82)
    return image_rel, thumb_rel


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


def burst_paths(image_rel: str, images_dir: Path | None = None) -> list[str]:
    """A visit's saved burst frames, in capture order (empty when only the best frame was kept)."""
    images_dir = images_dir or config.IMAGES_DIR
    stem = Path(image_rel.removesuffix(".jpg"))
    found = (images_dir / stem).parent.glob(f"{stem.name}_b*.jpg")
    nums = sorted(int(n) for p in found if (n := p.stem.rpartition("_b")[2]).isdigit())
    return [f"{stem}_b{n}.jpg" for n in nums]


def frame_bytes(image_rel: str, thumb_rel: str, images_dir: Path | None = None) -> int:
    """Disk space used by one visit's photo, thumbnail and any burst frames (missing files count as 0)."""
    return frames_bytes([(image_rel, thumb_rel)], images_dir)[0]


def frames_bytes(visits: list[tuple[str, str]], images_dir: Path | None = None) -> list[int]:
    """frame_bytes for many (image, thumbnail) pairs at once, in the same order.

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
                        stem, _, n = e.name.removesuffix(".jpg").rpartition("_b")
                        if e.name.endswith(".jpg") and stem and n.isdigit():
                            bursts[stem] = bursts.get(stem, 0) + size
            except (FileNotFoundError, NotADirectoryError):
                pass
            folders[parent] = (sizes, bursts)
        return folders[parent]

    out = []
    for image_rel, thumb_rel in visits:
        total = 0
        for rel in (image_rel, thumb_rel):
            total += folder(rel)[0].get(Path(rel).name, 0)
        total += folder(image_rel)[1].get(Path(image_rel).name.removesuffix(".jpg"), 0)
        out.append(total)
    return out


def delete_frame(image_rel: str, thumb_rel: str, images_dir: Path | None = None) -> None:
    images_dir = images_dir or config.IMAGES_DIR
    for rel in (image_rel, thumb_rel, *burst_paths(image_rel, images_dir)):
        (images_dir / rel).unlink(missing_ok=True)

