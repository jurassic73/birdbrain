"""Small pictures for tests: a flat frame, its JPEG bytes, and the size of an image file."""
import cv2
import numpy as np


def image_size(path) -> tuple[int, int]:
    """An image file's (width, height), closing the file again."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def frame(h: int = 360, w: int = 640, color=(90, 110, 80)) -> np.ndarray:
    """A flat BGR frame (substream size by default; pass h=1080, w=1920 for a main-stream one)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    return img


def jpeg(img: np.ndarray | None = None, **frame_kwargs) -> bytes:
    """JPEG bytes of `img`, or of a flat frame built with `frame_kwargs`."""
    if img is None:
        img = frame(**frame_kwargs)
    return cv2.imencode(".jpg", img)[1].tobytes()
