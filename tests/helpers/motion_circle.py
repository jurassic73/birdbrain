"""Shared helpers for the motion circle tests (used by more than one test module)."""
import cv2
import numpy as np

from birdbrain import config, motion_circle as mc, zones


W, H = 1920, 1080


ZONES = zones.Zones(ring=(0.5, 0.55, 0.16, 0.12, 0.13, 0.095), crop=(0.25, 0.25, 0.5, 0.55),
                    ground=((0.05, 0.6), (0.95, 0.6), (0.95, 0.98), (0.05, 0.98)), ground_crop=(0.0, 0.5, 1.0, 0.5))


RIM_TOP = (0.5, 0.55 - 0.12)  # the far rim, where a small bird perches


def scene(seed: int = 0) -> np.ndarray:
    """An empty yard in gray: busy mulch everywhere, a pale rim and dark water."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(50, 190, (135, 240)).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    outer, inner = config.ring_ellipses(ZONES.ring)
    config.draw_ellipse(img, outer, 200)
    config.draw_ellipse(img, inner, 70)
    return img


def animal(img: np.ndarray, at, size=(60, 42), shade: int = 25) -> np.ndarray:
    out = img.copy()
    cv2.ellipse(out, (int(at[0] * W), int(at[1] * H)), size, 0, 0, 360, shade, -1)
    return out


def jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def circle(img: np.ndarray, empty: np.ndarray, trigger: str = "rim"):
    return mc.circle_for(jpeg(img), empty, ZONES, trigger, "stored")


def near(c: dict, at, tol: float = 0.02) -> bool:
    return abs(c["x"] - at[0]) < tol and abs(c["y"] - at[1]) < tol
