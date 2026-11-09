"""The always-on loop: read the substream, run the HEF on every frame, track detections into events, keep a ring
buffer of the last few seconds so an event's arrival frames can be saved, and write events, snapshots and a
heartbeat. Fails soft: a lost stream or a Hailo error is logged and retried, never fatal.

Shadow only (see the package docstring). The watcher is never touched: this process has its own RTSP session,
its own database file and its own snapshot folder.
"""

import logging
import os
import re
import time
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .. import config, zones
from . import geometry, store
from .engine import Engine

log = logging.getLogger("birdbrain.hailo")

NIGHT_CHECK_FRAMES = 15   # like the watcher: re-check IR mode about once a second
STALL_SECONDS = 30        # reconnect when no frame has arrived for this long
STREAM_TIMEOUT_MS = 10000
HEARTBEAT_EVERY = 5.0     # seconds between heartbeat file writes
HEARTBEAT_ROW_EVERY = 60.0
ZONES_EVERY = 60.0        # seconds between re-reads of the zones in force
RETRY_SECONDS = 10
PRE_SECONDS = (2.0, 1.0)  # arrival shots saved from the ring buffer, this long before the first detection
JPEG_QUALITY = 80


def redact(text: str) -> str:
    return re.sub(r"rtsp://[^@/\s]*@", "rtsp://***@", text)


def open_capture(url: str) -> cv2.VideoCapture:
    params = []
    for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        if hasattr(cv2, name):
            params += [getattr(cv2, name), STREAM_TIMEOUT_MS]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(url, cv2.CAP_FFMPEG)


class RingBuffer:
    """The last `seconds` of frames with their monotonic times, so the moments before an event can be saved."""

    def __init__(self, seconds: float, fps: float):
        self.frames: deque[tuple[float, np.ndarray]] = deque(maxlen=max(int(seconds * fps) + 2, 2))

    def push(self, t: float, frame: np.ndarray) -> None:
        self.frames.append((t, frame))

    def nearest(self, t: float) -> tuple[float, np.ndarray] | None:
        """The frame closest to time t, or None when the buffer is empty."""
        if not self.frames:
            return None
        return min(self.frames, key=lambda item: abs(item[0] - t))


def draw(frame: np.ndarray, det: geometry.Detection | None, note: str = "") -> np.ndarray:
    """A copy of the frame with the detection's box and label on it (for the owner's review, never for a model)."""
    out = frame.copy()
    h, w = out.shape[:2]
    if det is not None:
        x0, y0, x1, y1 = (int(round(v * s)) for v, s in zip(det.box, (w, h, w, h)))
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 1)
        cv2.putText(out, f"{det.label} {det.score:.2f}", (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 220, 255), 1, cv2.LINE_AA)
    if note:
        cv2.putText(out, note, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def cpu_percent(before: tuple[float, float], after: tuple[float, float]) -> float:
    """This process's CPU use between two (os.times user+system, wall) samples, in percent of one core."""
    (cpu0, wall0), (cpu1, wall1) = before, after
    return round(100 * (cpu1 - cpu0) / (wall1 - wall0), 1) if wall1 > wall0 else 0.0


class OpenEvent:
    def __init__(self, event_id: int, started_at: datetime, snapshots: list[str]):
        self.id, self.started_at, self.snapshots = event_id, started_at, snapshots
        self.best_frame: np.ndarray | None = None
        self.best_score = -1.0


class Service:
    def __init__(self):
        self.engine: Engine | None = None
        self.engine_failed_at = 0.0
        self.tracker = geometry.Tracker(min_hits=config.HAILO_MIN_HITS, gap=config.HAILO_EVENT_GAP)
        self.ring = RingBuffer(config.HAILO_RING_SECONDS, config.SUB_FPS / config.HAILO_EVERY)
        self.open: dict[int, OpenEvent] = {}  # track id -> its event row
        self.other_classes: Counter = Counter()  # non-animal detections per class, this run
        self.errors = 0      # Hailo failures this run (each reopens the device)
        self.reconnects = 0  # substream reconnects this run
        self.zones = zones.from_env()
        self.zones_read_at = 0.0
        self.night = False
        self.frames = 0
        self.fps = 0.0
        self.last_frame_t = 0.0
        self.last_frame_at: str | None = None
        self.started = time.monotonic()
        self._fps_window: deque[float] = deque(maxlen=150)
        self._cpu_sample = self._times()
        self._heartbeat_t = 0.0
        self._heartbeat_row_t = 0.0

    # --- helpers --------------------------------------------------------------------------------------------------

    @staticmethod
    def _times() -> tuple[float, float]:
        t = os.times()
        return t.user + t.system, time.monotonic()

    def wall(self, t_mono: float) -> datetime:
        """The wall-clock time of a monotonic instant, with the second resolution visits use."""
        return (datetime.now() - timedelta(seconds=time.monotonic() - t_mono)).replace(microsecond=0)

    def load_zones(self) -> None:
        """The zones in force (recalibrations live in the production database): read read-only, .env otherwise."""
        now = time.monotonic()
        if now - self.zones_read_at < ZONES_EVERY:
            return
        self.zones_read_at = now
        try:
            conn = store.production_db()
            try:
                self.zones = zones.load(conn)
            finally:
                conn.close()
        except Exception:
            self.zones = zones.from_env()
            log.warning("could not read the zones from the database; using .env", exc_info=True)

    def ensure_engine(self) -> bool:
        if self.engine is not None:
            return True
        if time.monotonic() - self.engine_failed_at < RETRY_SECONDS:
            return False
        try:
            self.engine = Engine(config.HAILO_HEF, config.HAILO_MIN_SCORE)
            return True
        except Exception:
            self.engine_failed_at = time.monotonic()
            log.exception("could not open the Hailo device; retrying in %d s", RETRY_SECONDS)
            return False

    def drop_engine(self) -> None:
        if self.engine is not None:
            self.engine.close()
        self.engine, self.engine_failed_at = None, time.monotonic()

    # --- events ---------------------------------------------------------------------------------------------------

    def open_event(self, tr: geometry.Track, frame: np.ndarray, t: float) -> None:
        started = self.wall(tr.first_t)
        row = dict(started_at=started.isoformat(), label=tr.label, score=tr.best.score if tr.best else None,
                   labels=dict(tr.labels), animal=tr.animal, zone=tr.zone, in_zone=tr.zone in geometry.IN_ZONE,
                   frames=tr.hits, box=list(tr.best.box) if tr.best else None, model=self.engine.model if self.engine else None,
                   fps=round(self.fps, 1), night=self.night, concurrent=tr.concurrent, track_id=tr.id)
        with store.session() as conn:
            event_id = store.insert_event(conn, **row)
        ev = OpenEvent(event_id, started, [])
        for back in PRE_SECONDS:  # the arrival: frames from before the first detection
            item = self.ring.nearest(tr.first_t - back)
            if item is not None and tr.first_t - item[0] > back / 2:
                rel = store.snapshot_rel(started, event_id, f"pre{int(back)}s")
                store.save_jpeg(rel, jpeg(draw(item[1], None, f"{back:.0f} s before")))
                ev.snapshots.append(rel)
        rel = store.snapshot_rel(started, event_id, "start")
        store.save_jpeg(rel, jpeg(draw(frame, tr.best, "first frames")))
        ev.snapshots.append(rel)
        ev.best_frame, ev.best_score = frame.copy(), tr.best.score if tr.best else 0.0
        self.open[tr.id] = ev
        with store.session() as conn:
            store.update_event(conn, event_id, snapshots=ev.snapshots)
        log.info("event %d: %s %.2f in the %s zone (%d frames so far)", event_id, tr.label, tr.best.score if tr.best else 0,
                 tr.zone, tr.hits)

    def follow_event(self, tr: geometry.Track, frame: np.ndarray) -> None:
        ev = self.open.get(tr.id)
        if ev is not None and tr.best is not None and tr.best.score > ev.best_score:
            ev.best_frame, ev.best_score = frame.copy(), tr.best.score

    def close_event(self, tr: geometry.Track, frame: np.ndarray | None) -> None:
        ev = self.open.pop(tr.id, None)
        if ev is None:
            return
        if ev.best_frame is not None:
            rel = store.snapshot_rel(ev.started_at, ev.id, "best")
            store.save_jpeg(rel, jpeg(draw(ev.best_frame, tr.best, "best frame")))
            ev.snapshots.append(rel)
        if frame is not None:
            rel = store.snapshot_rel(ev.started_at, ev.id, "end")
            store.save_jpeg(rel, jpeg(draw(frame, None, "after")))
            ev.snapshots.append(rel)
        with store.session() as conn:
            store.update_event(conn, ev.id, ended_at=self.wall(tr.last_t).isoformat(), label=tr.label,
                               score=tr.best.score if tr.best else None, labels=dict(tr.labels), animal=tr.animal,
                               zone=tr.zone, in_zone=tr.zone in geometry.IN_ZONE, frames=tr.hits,
                               seconds=round(tr.seconds, 1), box=list(tr.best.box) if tr.best else None,
                               snapshots=ev.snapshots, concurrent=tr.concurrent)
        log.info("event %d ended: %s %.2f, %d frames, %.1f s, %s zone", ev.id, tr.label,
                 tr.best.score if tr.best else 0, tr.hits, tr.seconds, tr.zone)
        try:
            store.prune_snapshots()
        except Exception:
            log.warning("could not prune the snapshot folder", exc_info=True)

    # --- the loop -------------------------------------------------------------------------------------------------

    def heartbeat(self, t: float, force: bool = False) -> None:
        if not force and t - self._heartbeat_t < HEARTBEAT_EVERY:
            return
        self._heartbeat_t = t
        sample = self._times()
        cpu = cpu_percent(self._cpu_sample, sample)
        self._cpu_sample = sample
        temp = self.engine.chip_temperature() if self.engine else None
        data = {
            "at": datetime.now().replace(microsecond=0).isoformat(),
            "last_frame_at": self.last_frame_at,
            "last_frame_age_s": round(t - self.last_frame_t, 1) if self.last_frame_t else None,
            "fps": round(self.fps, 1),
            "infer_ms": round(self.engine.infer_ms, 1) if self.engine else None,
            "cpu_pct": cpu,
            "chip_temp_c": temp,
            "frames": self.frames,
            "model": self.engine.model if self.engine else None,
            "view": config.HAILO_VIEW,
            "night": self.night,
            "errors": self.errors,
            "reconnects": self.reconnects,
            "open_events": len(self.open),
            "other_classes": dict(self.other_classes.most_common(5)),
            "uptime_s": int(t - self.started),
            "pid": os.getpid(),
        }
        try:
            store.write_heartbeat(data)
        except Exception:
            log.warning("could not write the heartbeat file", exc_info=True)
        if t - self._heartbeat_row_t >= HEARTBEAT_ROW_EVERY:
            self._heartbeat_row_t = t
            try:
                with store.session() as conn:
                    store.add_heartbeat(conn, at=data["at"], fps=data["fps"], infer_ms=data["infer_ms"], cpu_pct=cpu,
                                        chip_temp_c=temp, frames=self.frames, model=data["model"])
            except Exception:
                log.warning("could not write the heartbeat row", exc_info=True)

    def step(self, frame: np.ndarray, t: float) -> None:
        """One substream frame at monotonic time t."""
        self.frames += 1
        self.last_frame_t, self.last_frame_at = t, datetime.now().replace(microsecond=0).isoformat()
        self._fps_window.append(t)
        if len(self._fps_window) > 1:
            self.fps = (len(self._fps_window) - 1) / (self._fps_window[-1] - self._fps_window[0])
        if (self.frames - 1) % config.HAILO_EVERY:
            return
        if self.frames % NIGHT_CHECK_FRAMES == 1:
            self.night = geometry.frame_is_night(frame)
        self.load_zones()
        self.ring.push(t, frame)
        if not self.ensure_engine():
            self.heartbeat(t)
            return
        try:
            detections = self.detect(frame)
        except Exception:
            log.exception("Hailo inference failed; reopening the device")
            self.errors += 1
            self.drop_engine()
            self.heartbeat(t, force=True)
            return
        # Events are for animal-like classes only: at night the IR-lit bowl is a "toilet" at 0.6 for hours on end.
        # The other classes are tallied for the heartbeat, so what the scene itself is called stays visible.
        for d in detections:
            if not d.animal:
                self.other_classes[d.label] += 1
        started, ended = self.tracker.update([d for d in detections if d.animal], t, self.zones)
        for tr in started:
            self.open_event(tr, frame, t)
        for tr in self.tracker.tracks:
            if tr.confirmed and tr.last_t == t:
                self.follow_event(tr, frame)
        for tr in ended:
            self.close_event(tr, frame)
        self.heartbeat(t)

    def detect(self, frame: np.ndarray) -> list[geometry.Detection]:
        """Detections as fractions of the whole frame: on the crop box's cut-out (HAILO_VIEW=crop, when the zones
        have one) or on the whole frame."""
        crop = self.zones.crop if config.HAILO_VIEW == "crop" else None
        if crop:
            px = zones.pixel_box(frame.shape, crop)
            if px:
                x0, y0, x1, y1 = px
                return [geometry.to_frame(d, crop) for d in self.engine.infer(frame[y0:y1, x0:x1])]
        return self.engine.infer(frame)

    def lost_stream(self) -> None:
        for tr in self.tracker.flush():
            self.close_event(tr, None)

    def run(self) -> None:
        url = config.rtsp_url(config.RTSP_SUB)
        while True:
            cap = open_capture(url)
            if not cap.isOpened():
                log.warning("substream unavailable; retrying in %d s", RETRY_SECONDS)
                self.heartbeat(time.monotonic(), force=True)
                time.sleep(RETRY_SECONDS)
                continue
            log.info("watching %s with %s (view: %s)", redact(url), config.HAILO_HEF.name, config.HAILO_VIEW)
            last = time.monotonic()
            while time.monotonic() - last < STALL_SECONDS:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                last = time.monotonic()
                try:
                    self.step(frame, last)
                except Exception:
                    log.exception("couldn't process a frame; carrying on")
            log.warning("lost the substream; reconnecting")
            self.reconnects += 1
            cap.release()
            self.lost_stream()
            time.sleep(2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if config.HAILO_DETECT == "off":
        log.info("HAILO_DETECT=off; nothing to do")
        return
    if not Path(config.HAILO_HEF).exists():
        log.error("HEF not found: %s", config.HAILO_HEF)
        raise SystemExit(1)
    with store.session() as conn:
        conn.execute("SELECT 1")  # creates the shadow database and its tables
    service = Service()
    try:
        service.run()
    finally:
        service.lost_stream()
        service.drop_engine()
