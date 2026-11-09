"""The Hailo-8 side: one VDevice, one configured HEF, synchronous inference on a letterboxed frame, and the chip
temperature. HailoRT is imported lazily so the rest of the package (and its tests) never need it.

The ready-made HEFs in /usr/share/hailo-models carry their own NMS post-process (score 0.2, IoU 0.7 for yolov8s),
so the output is already boxes per class; see geometry.parse_nms.
"""

import logging
import time
from pathlib import Path

import numpy as np

from . import geometry

log = logging.getLogger("birdbrain.hailo")


class Engine:
    """Open with Engine(hef_path); use infer(bgr) and chip_temperature(); close() releases the device."""

    def __init__(self, hef_path: Path, min_score: float):
        from hailo_platform import (ConfigureParams, FormatType, HailoStreamInterface, HEF, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams, VDevice)

        self.hef_path, self.min_score = Path(hef_path), min_score
        self.model = self.hef_path.stem
        start = time.perf_counter()
        self.vdevice = VDevice(VDevice.create_params())
        # Taken now: once the pipeline is active, asking the VDevice for its devices fails, and this handle keeps
        # answering temperature reads while inference runs.
        self.devices = self.vdevice.get_physical_devices()
        hef = HEF(str(self.hef_path))
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        self.group = self.vdevice.configure(hef, cfg)[0]
        self.in_name = hef.get_input_vstream_infos()[0].name
        self.out_name = hef.get_output_vstream_infos()[0].name
        self.size = int(hef.get_input_vstream_infos()[0].shape[0])
        in_params = InputVStreamParams.make(self.group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(self.group, format_type=FormatType.FLOAT32)
        self.pipe = InferVStreams(self.group, in_params, out_params)
        self.pipe.__enter__()
        self.active = self.group.activate(self.group.create_params())
        self.active.__enter__()
        self.infer_ms = 0.0  # a running average of the last inferences, for the heartbeat
        log.info("Hailo-8 ready with %s (%dx%d) in %.2f s", self.model, self.size, self.size, time.perf_counter() - start)

    def infer(self, bgr: np.ndarray) -> list[geometry.Detection]:
        """Detections on one BGR frame, best first, at or above min_score."""
        batch, ratio = geometry.letterbox(bgr, self.size)
        start = time.perf_counter()
        out = self.pipe.infer({self.in_name: batch})[self.out_name]
        ms = (time.perf_counter() - start) * 1000
        self.infer_ms = ms if not self.infer_ms else 0.9 * self.infer_ms + 0.1 * ms
        return geometry.parse_nms(out[0], ratio, bgr.shape, self.min_score, self.size)

    def chip_temperature(self) -> float | None:
        """The hotter of the chip's two sensors in C, read through the device this process already holds."""
        try:
            t = self.devices[0].control.get_chip_temperature()
            return round(max(t.ts0_temperature, t.ts1_temperature), 1)
        except Exception:
            return None

    def close(self) -> None:
        for step in (lambda: self.active.__exit__(None, None, None), lambda: self.pipe.__exit__(None, None, None),
                     self.vdevice.release):
            try:
                step()
            except Exception:
                pass
