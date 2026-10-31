import io
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image

from . import config

THUMB_PX = 400


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
) -> list[str]:
    """Save a whole burst next to its visit's photo as <stem>_b1.jpg, _b2.jpg... in capture order (no thumbnails).

    Kept for false triggers, so a doubtful one can be checked frame by frame. Returns the relative paths.
    """
    images_dir = images_dir or config.IMAGES_DIR
    stem = image_rel.removesuffix(".jpg")
    paths = []
    for i, jpeg in enumerate(frames, 1):
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if max_edge:
            img.thumbnail((max_edge, max_edge))
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
    images_dir = images_dir or config.IMAGES_DIR
    total = 0
    for rel in (image_rel, thumb_rel, *burst_paths(image_rel, images_dir)):
        try:
            total += (images_dir / rel).stat().st_size
        except FileNotFoundError:
            pass
    return total


def delete_frame(image_rel: str, thumb_rel: str, images_dir: Path | None = None) -> None:
    images_dir = images_dir or config.IMAGES_DIR
    for rel in (image_rel, thumb_rel, *burst_paths(image_rel, images_dir)):
        (images_dir / rel).unlink(missing_ok=True)

