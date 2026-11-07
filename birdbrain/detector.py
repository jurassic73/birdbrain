"""Local animal detector (YOLOX-tiny, Apache-2.0, COCO classes) used as a cheap first look before Claude.

It only answers "is there an animal or person here?". COCO has no squirrel, raccoon or possum class, so
an empty verdict is a hint, not proof; see DETECTOR in config for how much it is trusted.
CLI: python -m birdbrain.detector photo.jpg
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config

log = logging.getLogger("birdbrain.detector")

COCO = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
    "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie "
    "suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
    "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
    "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush"
).split()
# Anything that should still go to Claude. Small mammals often come out as cat, dog or bear.
VISITOR_CLASSES = {"person", "bird", "cat", "dog", "horse", "sheep", "cow", "bear", "teddy_bear"}
INPUT = 416
STRIDES = (8, 16, 32)


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the source image


def preprocess(bgr: np.ndarray, size: int = INPUT) -> tuple[np.ndarray, float]:
    """Letterbox to size x size on gray (114), as YOLOX was trained; returns NCHW float32 and the scale."""
    ratio = min(size / bgr.shape[0], size / bgr.shape[1])
    resized = cv2.resize(bgr, (int(bgr.shape[1] * ratio), int(bgr.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), ratio


def _grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for s in STRIDES:
        n = size // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids), np.concatenate(strides)


OBJECT_MIN = 0.05  # objectness kept for the scenery check (birdbrain.scenery), whatever the class


def objects(output: np.ndarray, ratio: float, shape: tuple[int, int], size: int = INPUT, iou: float = 0.45) -> list[list[float]]:
    """Class-agnostic boxes: [objectness, x, y, w, h] with the box as fractions of the image, after NMS.
    A squirrel scores as an object even though COCO has no squirrel class."""
    out = output.reshape(-1, 85).astype(np.float32).copy()
    grids, strides = _grids(size)
    out[:, :2] = (out[:, :2] + grids) * strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides
    out = out[out[:, 4] >= OBJECT_MIN]
    if not len(out):
        return []
    boxes = np.stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 2], out[:, 3]], 1) / ratio
    h, w = shape
    found = []
    for i in np.array(cv2.dnn.NMSBoxes(boxes.tolist(), out[:, 4].tolist(), OBJECT_MIN, iou)).flatten():
        x, y, bw, bh = boxes[i]
        found.append([round(float(out[i, 4]), 3), *(round(float(v), 4) for v in (x / w, y / h, bw / w, bh / h))])
    return sorted(found, reverse=True)


def decode(output: np.ndarray, ratio: float, min_score: float, size: int = INPUT, iou: float = 0.45) -> list[Detection]:
    """YOLOX raw output (N, 85: cx, cy, w, h, objectness, 80 class scores) to boxes after NMS."""
    out = output.reshape(-1, 85).astype(np.float32).copy()
    grids, strides = _grids(size)
    out[:, :2] = (out[:, :2] + grids) * strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides
    class_scores = out[:, 4:5] * out[:, 5:]
    cls = class_scores.argmax(1)
    score = class_scores[np.arange(len(cls)), cls]
    keep = score >= min_score
    if not keep.any():
        return []
    out, cls, score = out[keep], cls[keep], score[keep]
    boxes = np.stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 2], out[:, 3]], 1) / ratio
    idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), score.tolist(), cls.tolist(), min_score, iou)
    found = []
    for i in np.array(idx).flatten():
        x, y, w, h = boxes[i]
        found.append(Detection(COCO[cls[i]], float(score[i]), (int(x), int(y), int(x + w), int(y + h))))
    return sorted(found, key=lambda d: d.score, reverse=True)


class Detector:
    def __init__(self, model_path: Path):
        import onnxruntime as ort  # imported here so the web app never pays for it

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # leave the Pi's other cores to the watcher and other services
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, bgr: np.ndarray, min_score: float) -> list[Detection]:
        return self.detect_all(bgr, min_score)[0]

    def detect_all(self, bgr: np.ndarray, min_score: float) -> tuple[list[Detection], list[list[float]]]:
        """Class detections plus the class-agnostic objects, from one inference."""
        tensor, ratio = preprocess(bgr)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return decode(output, ratio, min_score), objects(output, ratio, bgr.shape[:2])


def verdict(detections: list[Detection], seconds: float, objs: list[list[float]] = ()) -> dict:
    """What gets stored with the visit and shown on its page."""
    visitors = [d for d in detections if d.label in VISITOR_CLASSES]
    return {
        "objects": list(objs)[:20],
        "animal": bool(visitors),
        "best": {"label": visitors[0].label, "score": round(visitors[0].score, 3)} if visitors else None,
        "found": [{"label": d.label, "score": round(d.score, 3)} for d in detections[:5]],
        "seconds": round(seconds, 2),
    }


_detector: Detector | None = None


def run(jpeg: bytes, crop=None) -> dict | None:
    """Detector verdict for one frame, or None when the detector is off or its model is missing."""
    global _detector
    if config.DETECTOR == "off" or not config.DETECTOR_MODEL.exists():
        return None
    if _detector is None:
        _detector = Detector(config.DETECTOR_MODEL)
    from .watch import roi  # local import: watch imports this module

    img = roi(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), crop)
    if img is None:
        return None  # the crop box lies outside the frame: nothing to look at
    start = time.perf_counter()
    found, objs = _detector.detect_all(img, config.DETECTOR_MIN_SCORE)
    return verdict(found, time.perf_counter() - start, objs)


# --- finding the animal for the tight look (birdbrain.finder) ------------------------------------------------

FINDER_CLASSES = VISITOR_CLASSES - {"person", "teddy_bear"}  # a person is never the animal the tight look wants


def to_frame_box(det_box, crop_px, shape) -> list[float]:
    """A detection's (x0, y0, x1, y1) pixel box inside the crop (crop_px: the crop's (x0, y0, x1, y1) on the frame,
    None for the whole frame) as [x, y, w, h] fractions of a frame of `shape` (H, W, ...), held inside the crop."""
    H, W = shape[:2]
    cx0, cy0, cx1, cy1 = crop_px if crop_px else (0, 0, W, H)
    x0, y0, x1, y1 = det_box
    x0, x1 = (min(max(float(v), 0.0), float(cx1 - cx0)) for v in (x0, x1))
    y0, y1 = (min(max(float(v), 0.0), float(cy1 - cy0)) for v in (y0, y1))
    return [round((cx0 + x0) / W, 4), round((cy0 + y0) / H, 4), round((x1 - x0) / W, 4), round((y1 - y0) / H, 4)]


def animal_box(jpeg: bytes, crop, min_score: float) -> tuple[list[float] | None, dict]:
    """One YOLOX-tiny inference on the crop box (like `run`), for birdbrain.finder: the box of the highest-scoring
    detection of an animal class (FINDER_CLASSES: never a person), as full-frame fractions [x, y, w, h], plus
    {"label", "score", "seconds"}. (None, {"why": "no model"}) without a model file, (None, {"why": "nothing",
    "seconds"}) when it saw no animal, (None, {"why": "error"}) when anything failed. Independent of
    config.DETECTOR (the retired watcher shadow mode)."""
    global _detector
    if not config.DETECTOR_MODEL.exists():
        return None, {"why": "no model"}
    start = time.perf_counter()
    try:
        if _detector is None:
            _detector = Detector(config.DETECTOR_MODEL)
        from .watch import roi  # local import: watch imports this module
        from . import zones

        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        img = roi(frame, crop) if frame is not None else None
        if img is None:
            return None, {"why": "nothing", "seconds": round(time.perf_counter() - start, 3)}
        crop_px = zones.pixel_box(frame.shape, crop) if crop else None
        found = _detector.detect(img, min_score)
    except Exception:
        log.exception("the object detector failed to find the animal")
        return None, {"why": "error", "seconds": round(time.perf_counter() - start, 3)}
    seconds = round(time.perf_counter() - start, 3)
    best = next((d for d in found if d.label in FINDER_CLASSES), None)
    if best is None:
        return None, {"why": "nothing", "seconds": seconds}
    return to_frame_box(best.box, crop_px, frame.shape), {"label": best.label, "score": round(best.score, 3),
                                                          "seconds": seconds}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m birdbrain.detector photo.jpg [more.jpg ...]")
    det = Detector(config.DETECTOR_MODEL)
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        start = time.perf_counter()
        found = det.detect(img, config.DETECTOR_MIN_SCORE)
        print(path, verdict(found, time.perf_counter() - start))


if __name__ == "__main__":
    main()
