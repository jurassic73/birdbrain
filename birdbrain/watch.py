"""Watch the camera for motion and record visits.

python -m birdbrain.watch          run forever
python -m birdbrain.watch --once   capture and identify one frame now, then exit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402  (must come after the capture options are set)
import numpy as np  # noqa: E402

from . import (alert, autocal, classifier, config, db, identify, images, motion_circle, presence,  # noqa: E402
               retention, species_info, stay, weather, zones)

log = logging.getLogger("birdbrain.watch")

WARMUP_FRAMES = 30  # let the background model settle after (re)connecting
NIGHT_CHECK_FRAMES = 15  # how often (substream frames, ~1 s) to re-check whether the camera is in IR mode
STALL_SECONDS = 30  # reconnect when the substream hasn't delivered a frame for this long
STREAM_TIMEOUT_MS = 10000  # FFmpeg's open and read timeouts, so a stalled stream returns instead of blocking
last_visit_id: int | None = None  # set by handle_event, so the watcher can keep that visit open while it stays
# Cooldowns, backoffs, stays and the stream watchdog all run on time.monotonic(): the Pi has no hardware clock
# and NTP steps the wall clock after boot, which would stretch a wait or hold a trigger. time.time() and
# datetime.now() are only for timestamps that are stored or shown.



_push: subprocess.Popen | None = None
_push_started = 0.0
PUSH_TIMEOUT = 330  # seconds: the script stops itself after 300 (timeout); this is the backstop if it can't


def push_public() -> None:
    """Send the new visit to the public mirror now instead of waiting for cron (PUBLIC_DEST in .env).
    Runs in the background; skipped if the previous push is still going (the next one catches up), unless
    that one has hung past PUSH_TIMEOUT: then it's stopped so pushes never stay blocked behind it."""
    global _push, _push_started
    if not os.environ.get("PUBLIC_DEST"):
        return
    if _push is not None and _push.poll() is None:
        if time.monotonic() - _push_started < PUSH_TIMEOUT:
            return
        log.warning("public push still running after %d s: stopping it", PUSH_TIMEOUT)
        _push.terminate()
        try:
            _push.wait(5)
        except subprocess.TimeoutExpired:
            _push.kill()
            _push.wait()
    script = Path(__file__).resolve().parent.parent / "scripts" / "push_public.sh"
    if script.exists():
        _push = subprocess.Popen([str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _push_started = time.monotonic()

def redact(text: str) -> str:
    return re.sub(r"rtsp://[^@/\s]*@", "rtsp://***@", text)


def roi(frame: np.ndarray, box) -> np.ndarray | None:
    """The box's part of the frame (all of it without a box), or None when the box lies outside the frame."""
    if not box:
        return frame
    px = zones.pixel_box(frame.shape, box)
    if px is None:
        return None
    x0, y0, x1, y1 = px
    return frame[y0:y1, x0:x1]


def motion_mask(shape: tuple[int, int], ring, spots=()) -> np.ndarray:
    """255 on the rim (inside the outer ellipse, outside the water) plus any extra spots, e.g. rocks in the water."""
    outer, inner = config.ring_ellipses(ring)
    mask = np.zeros(shape, np.uint8)
    config.draw_ellipse(mask, outer, 255)
    config.draw_ellipse(mask, inner, 0)
    for s in spots:
        config.draw_ellipse(mask, s, 255)
    return mask


def water_mask(shape: tuple[int, int], ring, spots=()) -> np.ndarray:
    """255 on the open water: the ring's inner ellipse, minus the rock spots (those belong to the rim zone)."""
    _, inner = config.ring_ellipses(ring)
    mask = np.zeros(shape, np.uint8)
    config.draw_ellipse(mask, inner, 255)
    for s in spots:
        config.draw_ellipse(mask, s, 0)
    return mask


class WaterWatch:
    """Bird-in-the-water detector that ignores ripples.

    Ripples and raindrops make faint, scattered, flickering change. A bird makes strong change in one
    compact blob that stays roughly in place. So: strict diff threshold, speckle removed, and a trigger only
    when exactly one bird-sized, solid blob persists for several consecutive frames near the same spot.
    """

    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.area = max(cv2.countNonZero(mask), 1)
        ys, xs = np.nonzero(mask)
        self.span = max(int(np.ptp(xs)), int(np.ptp(ys)), 1) if len(xs) else 1
        self.streak = 0
        self.last: tuple[float, float] | None = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def blob(self, diff: np.ndarray) -> tuple[float, float] | None:
        """Centroid of the single qualifying blob in this frame, or None."""
        strong = cv2.threshold(diff, config.WATER_DIFF, 255, cv2.THRESH_BINARY)[1]
        strong = cv2.bitwise_and(strong, self.mask)
        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, self.kernel)   # drop speckle (ripples, drops)
        strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, self.kernel)  # knit a bird's body together
        contours, _ = cv2.findContours(strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        good = []
        for c in contours:
            area = cv2.contourArea(c) / self.area
            if not config.WATER_MIN_BLOB <= area <= config.WATER_MAX_BLOB:
                continue
            hull = cv2.contourArea(cv2.convexHull(c)) or 1
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) / hull < config.WATER_MIN_SOLIDITY or max(w, h) > 3.5 * max(min(w, h), 1):
                continue
            m = cv2.moments(c)
            good.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
        return good[0] if len(good) == 1 else None

    def update(self, diff: np.ndarray) -> bool:
        """Feed one frame's difference image; True once a blob has persisted long enough."""
        center = self.blob(diff)
        if center is None:
            self.streak, self.last = 0, None
            return False
        near = self.last is not None and np.hypot(center[0] - self.last[0], center[1] - self.last[1]) <= 0.25 * self.span
        self.streak = self.streak + 1 if near or self.last is None else 1
        self.last = center
        return self.streak >= config.WATER_PERSIST_FRAMES

    def reset(self) -> None:
        self.streak, self.last = 0, None


def ground_mask(shape: tuple[int, int], polygon, ring=None) -> np.ndarray:
    """255 inside the ground polygon, minus the birdbath itself (the rim and water zones watch that)."""
    H, W = shape
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.array([(int(x * W), int(y * H)) for x, y in polygon], np.int32)], 255)
    if ring:
        config.draw_ellipse(mask, config.ring_ellipses(ring)[0], 0)
    return mask


class GroundWatch:
    """Critters passing by on the ground.

    Wind-blown leaves and rain splashes are small and flicker; light changes (clouds, the dusk IR switch) hit
    the whole zone at once. An animal is one sizable blob that stays in view and moves at walking pace. At
    night the blob must be cat-sized: nothing small flies in after dark, and moths at the IR lights are the
    usual false alarm.
    """

    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.width = mask.shape[1]
        self.streak = 0
        self.last: tuple[float, float] | None = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.knit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def blob(self, diff: np.ndarray, night: bool) -> tuple[float, float] | None:
        """Centroid of the biggest animal-sized blob in this frame, or None."""
        strong = cv2.threshold(diff, config.GROUND_DIFF, 255, cv2.THRESH_BINARY)[1]
        strong = cv2.bitwise_and(strong, self.mask)
        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, self.kernel)  # drop leaf flicker and splashes
        strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, self.knit)   # knit a walking body together
        if cv2.countNonZero(strong) / strong.size > config.GROUND_MAX_BLOB:
            return None  # the light changed, not an animal
        low = config.GROUND_MIN_BLOB_NIGHT if night else config.GROUND_MIN_BLOB
        contours, _ = cv2.findContours(strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sized = [c for c in contours if low <= cv2.contourArea(c) / strong.size <= config.GROUND_MAX_BLOB]
        if not sized:
            return None
        m = cv2.moments(max(sized, key=cv2.contourArea))
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

    def update(self, diff: np.ndarray, night: bool) -> bool:
        """Feed one frame's difference image; True once a blob has persisted long enough."""
        center = self.blob(diff, night)
        if center is None:
            self.streak, self.last = 0, None
            return False
        near = self.last is None or np.hypot(center[0] - self.last[0], center[1] - self.last[1]) <= (
            config.GROUND_MAX_STEP * self.width)
        self.streak = self.streak + 1 if near else 1
        self.last = center
        return self.streak >= config.GROUND_PERSIST_FRAMES

    def reset(self) -> None:
        self.streak, self.last = 0, None


class MotionDetector:
    def __init__(self, box, min_area: float, ring=None, spots=(), ground=()):
        self.box, self.min_area, self.ring, self.spots = (None if ring else box), min_area, ring, spots
        self.ground_zone = ground
        self.bg: np.ndarray | None = None  # whole frame; the rim score looks at its part of it
        self.mask: np.ndarray | None = None
        self.water: WaterWatch | None = None
        self.ground: GroundWatch | None = None
        # The water zone keeps its own, much slower background (~1%/frame vs 5%): a bird sitting still in the
        # bath would otherwise fade into the background before it had persisted long enough to count.
        self.water_bg: np.ndarray | None = None
        # The ground's background is slow too: a possum waddling across fades into a 5%/frame background.
        self.ground_bg: np.ndarray | None = None
        self.water_hit = False  # set by score(): a bird-like blob has persisted in the open water
        self.ground_hit = False  # set by score(): an animal-sized blob has persisted on the ground
        self.night = False      # the camera is in IR mode; re-checked every NIGHT_CHECK_FRAMES
        self.frames = 0
        self.full: np.ndarray | None = None       # last frame, blurred gray, for the stay check
        self.full_diff: np.ndarray | None = None  # its change versus the running background

    def use(self, z: zones.Zones) -> bool:
        """Switch to recalibrated zones; the masks and backgrounds are rebuilt from the next frame."""
        new = (None if z.ring else z.roi, z.ring, z.spots, z.ground)
        if new == (self.box, self.ring, self.spots, self.ground_zone):
            return False
        self.box, self.ring, self.spots, self.ground_zone = new
        self.mask = self.water = self.ground = None
        self.reset()
        return True

    def reset(self) -> None:
        self.bg = None
        self.water_bg = None
        self.ground_bg = None
        self.frames = 0
        if self.water:
            self.water.reset()
        if self.ground:
            self.ground.reset()

    def score(self, frame: np.ndarray) -> float:
        """Fraction of the watched region (rim and rocks) that changed versus the running background.

        Also updates water_hit from the stricter open-water rule when a ring is configured (daytime only:
        nothing flies in to land on the water after dark), and ground_hit when a ground zone is.
        """
        if self.frames % NIGHT_CHECK_FRAMES == 0:
            self.night = frame_is_night(frame)
        self.frames += 1
        full = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        gray = self.part(full)
        if self.ring and (self.mask is None or self.mask.shape != gray.shape):
            self.mask = motion_mask(gray.shape, self.ring, self.spots)
            self.water = WaterWatch(water_mask(gray.shape, self.ring, self.spots)) if config.WATER_WATCH else None
        if self.ground_zone and (self.ground is None or self.ground.mask.shape != full.shape):
            self.ground = GroundWatch(ground_mask(full.shape, self.ground_zone, self.ring))
        self.water_hit = self.ground_hit = False
        if self.bg is None:
            self.bg = full.astype("float32")
            self.water_bg = gray.astype("float32")
            self.ground_bg = full.astype("float32")
            return 0.0
        full_diff = cv2.absdiff(full, cv2.convertScaleAbs(self.bg))
        self.full, self.full_diff = full, full_diff
        cv2.accumulateWeighted(full, self.bg, 0.05)
        diff = self.part(full_diff)
        if self.water is not None:
            if self.night:
                self.water.reset()
            else:
                self.water_hit = self.water.update(cv2.absdiff(gray, cv2.convertScaleAbs(self.water_bg)))
            cv2.accumulateWeighted(gray, self.water_bg, 0.01)
        if self.ground is not None:
            self.ground_hit = self.ground.update(cv2.absdiff(full, cv2.convertScaleAbs(self.ground_bg)), self.night)
            cv2.accumulateWeighted(full, self.ground_bg, config.GROUND_BG_RATE)
        changed = cv2.dilate(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        if self.mask is not None:
            return cv2.countNonZero(cv2.bitwise_and(changed, self.mask)) / max(cv2.countNonZero(self.mask), 1)
        return cv2.countNonZero(changed) / changed.size

    def part(self, img: np.ndarray) -> np.ndarray:
        """The watched box's part of a frame, or the whole frame when the box lies outside it (a bad hand
        calibration; a recalibration never pushes one out)."""
        r = roi(img, self.box)
        return img if r is None else r

    def stay_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Everywhere a visitor can be, in full-frame coordinates: rim, rocks, water and ground."""
        if self.ring:
            mask = motion_mask(shape, self.ring, self.spots) | water_mask(shape, self.ring, self.spots)
        else:
            mask = stay.box_mask(shape, self.box)
        if self.ground_zone:
            mask |= ground_mask(shape, self.ground_zone, self.ring)
        return mask

    def empty_scene(self) -> np.ndarray | None:
        """The running background: the scene just before whatever triggered (a visitor fades in at 5%/frame)."""
        return None if self.bg is None else cv2.convertScaleAbs(self.bg)

    def scene(self) -> Scene | None:
        """The empty scene with where visitors can be and the open water, for counting animals in the photo."""
        empty = self.empty_scene()
        if empty is None:
            return None
        water = water_mask(empty.shape, self.ring, self.spots) if self.ring else None
        return Scene(empty, self.stay_mask(empty.shape), water)


class Scene(NamedTuple):
    """The scene just before a trigger (blurred gray substream frame) and its masks, for stay.count_animals."""
    empty: np.ndarray
    mask: np.ndarray
    water: np.ndarray | None


def count_animals(scene: Scene | None, jpeg: bytes, crop) -> int | None:
    """Fail-soft: separate visitor-sized changes in the photo's crop box (None when it can't tell)."""
    if scene is None:
        return None
    try:
        return stay.count_animals(jpeg, scene.empty, scene.mask, crop, scene.water)
    except Exception:
        log.exception("counting animals failed")
        return None


def sharpness(jpeg: bytes, crop=None) -> float:
    """Laplacian variance, measured only inside the crop: busy mulch around the bath shouldn't win."""
    patch = roi(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE), crop)
    return 0.0 if patch is None else float(cv2.Laplacian(patch, cv2.CV_64F).var())


def best_frames(frames: list[bytes], crop=None, n: int = 3) -> list[bytes]:
    """The burst's sharpest frames inside the birdbath, best first."""
    return sorted(frames, key=lambda f: sharpness(f, crop), reverse=True)[:n]


def frame_is_night(bgr: np.ndarray) -> bool:
    """IR night frames are effectively grayscale, so color saturation is near zero."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < 12


def is_night(jpeg: bytes) -> bool:
    return frame_is_night(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))


def open_capture(url: str) -> cv2.VideoCapture:
    """The substream, with FFmpeg's open and read timeouts set when this OpenCV build has them, so a stalled
    stream returns from read() in STREAM_TIMEOUT_MS instead of blocking it for half a minute."""
    params = []
    for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        if hasattr(cv2, name):
            params += [getattr(cv2, name), STREAM_TIMEOUT_MS]
    if params:
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def capture_burst() -> list[bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
            "-i", config.rtsp_url(config.RTSP_MAIN), "-vf", f"fps={config.BURST_FPS}", "-frames:v", str(config.BURST_FRAMES),
            "-q:v", "2", f"{tmp}/f%02d.jpg",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            log.warning("burst capture timed out")
            return []
        if proc.returncode != 0:
            log.warning("burst capture failed: %s", redact(proc.stderr.strip())[:300])
        return [p.read_bytes() for p in sorted(Path(tmp).glob("f*.jpg"))]


def calls_today(ground: bool = False) -> int:
    """Visits today that Claude was asked about (visits the Pi settled cost nothing), counted
    separately for the ground zone so passers-by can't use up the birdbath's budget."""
    today = date.today()
    with db.session() as conn:
        return conn.execute(
            """SELECT COUNT(*) FROM visits WHERE captured_at >= ? AND captured_at < ? AND model LIKE 'claude%'
                 AND (COALESCE(json_extract(raw_json, '$.trigger'), '') = 'ground') = ?""",
            (today.isoformat(), (today + timedelta(days=1)).isoformat(), int(ground)),
        ).fetchone()[0]


def daily_limit(ground: bool) -> int:
    return config.GROUND_MAX_CALLS_PER_DAY if ground else config.MAX_CALLS_PER_DAY


def next_cooldown(visitor: str | None, false_streak: int) -> tuple[int, int]:
    """(seconds to wait, new false-trigger streak). Each false trigger in a row doubles the wait, so rain
    or wind backs off to COOLDOWN_NONE_MAX instead of calling Claude every 20 s; a real visitor resets it."""
    if visitor is not None and visitor != "none":
        return config.COOLDOWN_VISIT, 0
    streak = false_streak + 1
    return config.false_trigger_wait(streak), streak




def after_false(now: float, streak: int, cooldown: int, can_probe: bool) -> tuple[float, float]:
    """(cooldown_until, backoff_until) after a false trigger, the streak-th in a row. During the backoff,
    triggers are probes every COOLDOWN_NONE (with the local animal check). From PROBE_AFTER in a row the backoff
    runs FULL_CHECK_EVERY, so steady rain or blowing leaves cost one Claude call per half hour, not one
    every COOLDOWN_NONE_MAX (about 240 a night on 2026-10-26). can_probe must mean a trained check with a real
    verdict (probe_ready): with nothing filtering them, probes every COOLDOWN_NONE are 180 Claude calls an hour."""
    if not can_probe:
        return now + cooldown, now + cooldown
    backoff = config.FULL_CHECK_EVERY if streak >= config.PROBE_AFTER else cooldown
    return now + config.COOLDOWN_NONE, now + backoff


def judged(check: dict | None) -> bool:
    """Whether the local animal check actually judged the frame: a trained model gave it a score."""
    return bool(check) and check.get("score") is not None


def probe_passes(check: dict | None) -> bool:
    """A rain-backoff probe goes on to Claude only when the local animal check judged it and didn't call it
    empty. No verdict (the check off, not trained yet, or failed) is not a pass: nothing was filtering the
    probe, so it is dropped and the backoff goes back to plain doubling (PROBE_UNCHECKED)."""
    return judged(check) and not check.get("empty")


PROBE_EMPTY = "probe-empty"  # a backoff probe the local animal check found empty in: not stored, no Claude call
PROBE_UNCHECKED = "probe-unchecked"  # a probe with no verdict to filter it: not stored, no Claude call; the
                                     # backoff doubles as if there were no check
UNANSWERED = "unanswered"    # Claude couldn't answer (an API error): stored as "other" for review, so neither a
                             # visit (no stay, no alert) nor a false trigger (the rain backoff streak is left alone)

_last_logged: dict[str, float] = {}


def log_rarely(key: str, msg: str, *args, every: float = 60.0) -> None:
    """log.exception at most once per `every` seconds per key: a locked database during a rain run would
    otherwise fill the log on every frame."""
    now = time.monotonic()
    if now - _last_logged.get(key, -every) >= every:
        _last_logged[key] = now
        log.exception(msg, *args)


def fail_soft(what: str, fn, *args) -> None:
    """Run one side effect of a stored visit; a failure is logged, never allowed to turn the visit into a
    false trigger (the visit is already saved and paid for by the time these run)."""
    try:
        fn(*args)
    except Exception:
        log.exception("%s failed", what)


def streak_hit(motion: float) -> bool:
    return motion >= config.MOTION_MIN_AREA


_presence_cache: tuple[float, str, str] = (0.0, "", "")  # (read at, .env default, mode)


def presence_mode() -> str:
    """The local animal check's mode from Settings (else .env), re-read at most every 15 s."""
    global _presence_cache
    read_at, default, mode = _presence_cache
    if time.monotonic() - read_at > 15 or default != config.PRESENCE:
        with db.session() as conn:
            mode = presence.mode(conn)
        _presence_cache = (time.monotonic(), config.PRESENCE, mode)
    return mode


_hold_cache: tuple[float, bool] = (0.0, False)  # (read at, held)


def presence_held() -> bool:
    """Whether a `--backfill --redo` is holding the animal check (presence.held), re-read at most every 15 s like
    the mode."""
    global _hold_cache
    read_at, held = _hold_cache
    if time.monotonic() - read_at > 15:
        with db.session() as conn:
            held = presence.held(conn)
        _hold_cache = (time.monotonic(), held)
    return held


def probe_ready() -> bool:
    """Probes need a working local animal check: the embedding model on (CLASSIFIER) and loadable, the check not
    off, not held by a redo (its model is then trained on part of the frames, so a probe it called empty could be
    a bird: as with no model, probes wait), and trained on enough of this yard's frames to give a verdict. Without all of that a rain run just
    waits out its doubling backoff: unfiltered probes every COOLDOWN_NONE would ask Claude 180 times an hour
    and use up the day's budget before lunch. Fail-soft: if the mode can't be read (a locked database), there
    are no probes until it can be, and the caller carries on."""
    if not classifier.usable() or not presence.trained():
        return False
    try:
        return presence_mode() != "off" and not presence_held()
    except Exception:
        log_rarely("presence_mode", "couldn't read the animal check's mode; no probes until it can be read")
        return False


def _held_now() -> bool:
    """presence_held, fail-soft the safe way: if it can't be read, a probe counts as held (unchecked)."""
    try:
        return presence_held()
    except Exception:
        log_rarely("presence_hold", "couldn't read whether a redo holds the animal check; probes count as unchecked")
        return True


def motion_action(now: float, cooldown_until: float, backoff_until: float, can_probe: bool) -> str:
    """What motion should do right now: "wait", "probe" (during rain backoff: local animal check first, Claude
    only if it sees an animal) or "full" (normal capture and identify)."""
    if now < cooldown_until:
        return "wait"
    if now < backoff_until:
        return "probe" if can_probe else "wait"
    return "full"


def _scored(jpeg: bytes, z, trigger: str) -> tuple[np.ndarray | None, dict | None]:
    """The Pi animal check on one frame: its view of the zones in use for this trigger (classifier.presence_patch:
    the bowl square, or the padded ground crop), and the verdict, which records the view's kind."""
    emb = classifier.box_embedding(jpeg, z, trigger)
    kind = classifier.view_kind(z, trigger)
    with db.session() as conn:
        check = presence.check(conn, emb, view=kind)
    return emb, None if check is None else {**check, "view": kind}


def animal_check(frames: list[bytes], ranked: list[bytes], z, trigger: str = "rim"
                 ) -> tuple[np.ndarray | None, dict | None, list[bytes]]:
    """(embedding, verdict, ranked frames) from the local animal check, fail-soft (no verdict on any problem).
    `z` is the zones in use and `trigger` what fired, which together decide what the check looks at.

    It scores the sharpest frame and, when that looks empty and PRESENCE_LAST_FRAME is on, the burst's last frame
    too: a landing bird is usually in the bowl by then, while sharpness can favour ripples or mulch. The higher
    score counts (so the verdict is the max of the two), and its embedding is the one stored for training. When
    only the last frame shows an animal it becomes the photo, so Claude sees what the Pi saw."""
    try:
        emb, check = _scored(ranked[0], z, trigger)
    except Exception:
        log.exception("local animal check failed; asking Claude")
        return None, None, ranked
    last = frames[-1]
    if (not config.PRESENCE_LAST_FRAME or not check or check.get("score") is None or not check.get("empty")
            or last is ranked[0]):
        return emb, check, ranked
    try:
        emb2, check2 = _scored(last, z, trigger)
    except Exception:
        log.exception("animal check on the burst's last frame failed; going by the sharpest frame")
        return emb, check, ranked
    if not check2 or check2.get("score") is None:
        return emb, check, ranked
    scores = {"sharpest": check["score"], "last": check2["score"]}
    if check2["score"] <= check["score"]:
        return emb, {**check, "frames": scores, "frame": "sharpest"}, ranked
    if not check2["empty"]:
        log.info("animal check: only the burst's last frame shows an animal (%.3f vs %.3f); it becomes the photo",
                 check2["score"], check["score"])
        ranked = [last, *[f for f in ranked if f is not last]][:len(ranked)]
    return emb2, {**check2, "frames": scores, "frame": "last"}, ranked


def handle_event(motion: float | None = None, probe: bool = False, trigger: str = "rim",
                 scene: Scene | None = None) -> str | None:
    """Capture, identify and store one visit. Returns the visitor type, None if nothing was captured,
    PROBE_EMPTY for a backoff probe where the local animal check found nothing (discarded: rain shouldn't fill
    the disk or the bill), or UNANSWERED when Claude couldn't answer (the visit is still stored, for review).
    `scene` is the empty scene from the trigger, used to count the animals in view."""
    captured_at = datetime.now().replace(microsecond=0)
    frames = capture_burst()
    if not frames:
        return None
    z = zones.current()
    crop = z.ground_crop if trigger == "ground" else z.crop
    ranked = best_frames(frames, crop)
    # The local animal check: BioCLIP 2's embedding of a square around the bowl (ground triggers: the ground crop
    # padded to a square), scored by a model trained on this yard's labelled frames. It is the only gate: in gate
    # mode an empty verdict skips Claude (day and night; the owner reviews them), and a rain-backoff probe it calls
    # empty is discarded in any mode but off.
    emb = check = None
    pmode = presence_mode()
    if pmode != "off":
        emb, check, ranked = animal_check(frames, ranked, z, trigger)
    best = ranked[0]
    night = is_night(best)
    if probe and not judged(check):
        log.info("probe (%s trigger): the animal check gave no verdict; discarded, backing off without probes", trigger)
        return PROBE_UNCHECKED
    if probe and _held_now():
        log.info("probe (%s trigger): a redo holds the animal check; discarded, backing off without probes", trigger)
        return PROBE_UNCHECKED
    if probe and not probe_passes(check):
        log.info("probe (%s trigger): animal check found nothing; discarded", trigger)
        with db.session() as conn:
            db.bump(conn, "probe_empty")
        return PROBE_EMPTY
    empty = bool(check and check.get("empty"))
    guess = pick = looked = tight = None
    animals = None if empty else count_animals(scene, best, crop)
    crowded = bool(animals and animals > 1)
    crowd = None
    named = None  # which BioCLIP 2 look named the visit: "bowl" (the first look) or "tight"
    circle = found = None  # the circle from the motion, once it was looked for
    tight_vec = circle_vec = circle_model = None  # BioCLIP 2 embeddings of the tight squares, kept as training data
    haiku_look = []  # the embedding of the look on Haiku's circle, if one was made
    if not empty:  # the species gate: a sure daytime name of a species already settled here skips Claude
        with db.session() as conn:
            settled = db.settled_species(conn)
        guess, pick = classifier.gate_check(emb, night, settled)
        if guess:  # it reused the Pi animal check's embedding: say which view that was of
            guess = {**guess, "crop": classifier.view_kind(z, trigger)}
        if pick and crowded:
            # One BioCLIP guess names one animal; Claude counts them and names any other kind.
            log.info("species gate: %d animals in view, asking Claude instead of naming %s", animals, pick[0])
            crowd, pick = pick[0], None
        looked = guess  # the first look: kept below if it didn't name the visit (the Claude branch overwrites guess)
        if pick:
            named = "bowl"
        elif looked and not crowded and not night:
            # Rule H: the wide view is the bottleneck, so before asking Claude take one tight look at a square
            # around the patch that moved (about 6 s on a Pi 4), with a higher bar. No patch: Claude, as before.
            circle, found = motion_circle.live(best, scene.empty if scene else None, z, trigger,
                                               captured_at.isoformat(), night)
            if circle:
                (tight, pick), tight_vec = classifier.with_embedding(
                    classifier.tight_check, best, circle, crop, night, settled)
                if pick:
                    named = "tight"
                    log.info("species gate: the tight look named %s (%.2f); the first look had %s at %.2f", pick[0],
                             tight["top"][0]["score"], looked["top"][0]["name"] if looked.get("top") else "nothing",
                             looked["top"][0]["score"] if looked.get("top") else 0.0)
                elif tight:
                    log.info("species gate: the tight look did not name the visit; asking Claude")
            else:
                log.info("species gate: no patch of motion for a tight look (%s); asking Claude",
                         (found or {}).get("why") or "unknown")
    if pick:
        result = identify.settled_by_classifier(tight if named == "tight" else guess, pick, check)
        if named == "tight":
            # The tight look's square was around the circle from the motion: that circle is the visit's.
            result.raw["motion_circle"], result.raw["circle"] = found, circle
        elif config.CIRCLE_FROM_MOTION:
            # BioCLIP says what, not where: circle the biggest patch of change from the empty scene instead.
            circle, found = motion_circle.live(best, scene.empty if scene else None, z, trigger,
                                               captured_at.isoformat(), night)
            result.raw["motion_circle"] = found
            if circle:
                result.raw["circle"] = circle
    elif pmode == "gate" and empty:
        result = identify.settled_by_presence(check, None)
    else:
        with db.session() as conn:
            lessons = db.recent_lessons(conn)
            site = db.prompt_site(conn)
            confirmed = db.confirmed_species(conn)
        # By day the Pi's own guess on Haiku's circle can veto skipping Sonnet (not checked on infrared yet).

        def local_check(circle):
            guess, vec = classifier.with_embedding(classifier.run, best, circle, crop, night)
            haiku_look[:] = [vec]
            return guess

        if night:
            local_check = None
        try:
            result = identify.identify(best, captured_at, crop, lessons, extra_frames=ranked[1:], site=site,
                                       confirmed=confirmed, local_check=local_check, night=night)
        except Exception as e:
            # The API down, unreachable or timing out past the SDK's retries: a real bird may be in the photo, so
            # keep it as a visit for the review queue instead of losing it (and its burst) as a false trigger.
            log.exception("Claude couldn't answer; keeping the visit for review")
            result = identify.unanswered(e)
        if check:
            result.raw["presence"] = check
    unanswered = "error" in result.raw
    # When Claude answered, the classifier records its guess from Claude's circle for the Stats scorecard (already
    # made on Haiku's circle when it decided whether to ask Sonnet; again on Sonnet's circle if Sonnet answered).
    if not pick and result.sighting.visitor != "none":
        # The tight square around Claude's circle is also kept as training data: the look the stored guess is of.
        circle_model = "sonnet" if result.escalated else "haiku"
        if result.escalated or "classifier" not in result.raw:
            guess, circle_vec = classifier.with_embedding(classifier.run, best, result.raw.get("circle"), crop, night)
            if guess:
                result.raw["classifier"] = guess
            else:
                circle_vec = None  # no new guess: don't keep a vector the stored guess isn't of
        elif haiku_look:
            circle_vec = haiku_look[0]
    if looked and named != "bowl":
        # The BioCLIP 2 species check's first look did not name the visit: keep the guess and why, so the visit
        # page can show it before the tight look or Claude (the guess on Claude's circle above is the later
        # scorecard one; when the tight look named the visit, that look is calls.classifier).
        result.raw["classifier_gate"] = classifier.first_look(looked, settled, night, crowd)
    if tight and not pick:
        # The tight look did not name it either: kept with its reason and the circle it looked around, so the
        # visit page can show it after the first look. Claude was asked and reuses nothing from it.
        result.raw["classifier_tight"] = {**classifier.first_look(tight, settled, night, view="tight"), "circle": circle}
    if animals is not None:
        result.raw["animals_seen"] = animals  # scored against Claude's count later
    if crowd:
        result.raw["gate_crowd"] = {"species": crowd, "animals": animals}
    s = result.sighting
    false_trigger = s.visitor == "none"
    max_edge, quality = None, 85
    if false_trigger:
        with db.session() as conn:
            max_edge, quality = retention.PHOTO_SIZES[retention.photo_size(conn)]
    image_rel, thumb_rel = images.store_frame(best, captured_at, max_edge=max_edge, quality=quality)
    burst = None
    if false_trigger and len(frames) > 1:
        # The whole burst, in capture order, so a doubtful false trigger can be checked before and after.
        try:
            images.store_burst(frames, image_rel, max_edge=max_edge, quality=quality, captured_at=captured_at)
            burst = {"count": len(frames), "best": frames.index(best), "fps": config.BURST_FPS,
                     "sharpness": [round(sharpness(f, crop), 1) for f in frames]}
        except Exception:
            log.exception("saving the burst failed")
    alert_reason = None
    also = result.raw.get("also") or []
    # Every kind of animal in the photo with Claude's confidence in it: the main one first, then the others Claude
    # listed (nothing when Claude couldn't answer: "other" is a placeholder there, not a sighting to alert about).
    seen = [] if false_trigger or unanswered else [
        (s.visitor, s.species or s.visitor.capitalize(), s.confidence),
        *((a["visitor"], db.also_label(a), a.get("confidence") or "low") for a in also)]
    global last_visit_id
    with db.session() as conn:
        # Check BEFORE inserting so the new-species count doesn't include this visit yet.
        for visitor, label, confidence in seen:
            try:
                alert_reason = alert.reason(conn, visitor, label, captured_at, confidence, result.needs_review)
            except Exception:
                log.exception("alert check failed")
            if alert_reason:
                break
        visit_id = db.insert_visit(
            conn,
            captured_at=captured_at.isoformat(),
            image_path=image_rel,
            thumb_path=thumb_rel,
            visitor=s.visitor,
            species=s.species,
            count=s.count or None,
            confidence=s.confidence,
            notes=s.notes or None,
            model=result.model,
            escalated=int(result.escalated),
            needs_review=int(result.needs_review),
            is_night=int(night),
            also=json.dumps(also) if also else None,
            raw_json=json.dumps({"answer": s.model_dump(), "calls": result.raw, "motion": motion,
                                 "trigger": trigger, "probe": probe, "frames": len(frames),
                                 # The zones in use, so the visit page can draw the crop box and the zone that
                                 # fired as they were, even after recalibration moves them.
                                 "zones": z.to_dict(),
                                 **({"burst": burst} if burst else {})}),
        )
        if emb is not None:
            presence.store(conn, visit_id, emb)  # training data once its label is known (and corrected)
            if tight_vec is not None or circle_vec is not None:
                # Fingerprints of the tight views, for a species model of this yard later; never worth a visit.
                fail_soft("keeping the tight fingerprints", presence.store_tight, conn, visit_id, tight_vec,
                          circle_vec, circle_model if circle_vec is not None else None)
    last_visit_id = visit_id
    # From here on the visit is stored: whatever goes wrong below is logged, and the stored visitor still returned.
    if false_trigger:
        fail_soft("pruning false triggers", prune_false_triggers)
    # The photo budget: counts this visit and wakes its background check (cheap; any thinning happens there).
    fail_soft("noting the stored photo", retention.note_stored, image_rel, thumb_rel)
    if alert_reason:
        fail_soft("sending the alert", send_alert, visit_id, alert_reason, best)
    if not false_trigger:
        fail_soft("pushing to the public mirror", push_public)
    for _, label, _ in seen:
        fail_soft("prefetching the species card", species_info.prefetch, label)  # no-op once the card is cached
    det_note = "" if not check or check.get("score") is None else f", animal check {check['score']:.2f}"
    log.info("visit %d: %s %s (%s, %s%s%s)", visit_id, s.visitor, s.species or "", s.confidence,
             result.model, ", escalated" if result.escalated else "", det_note)
    return UNANSWERED if unanswered else s.visitor


def prune_false_triggers() -> None:
    with db.session() as conn:
        if pruned := retention.prune(conn):
            log.info("false-trigger storage limit: deleted the %d oldest", pruned)


def send_alert(visit_id: int, reason: str, photo: bytes) -> None:
    with db.session() as conn:
        topic = db.get_value(conn, "ntfy_topic") or ""
    click = f"{config.SITE_URL}/visit/{visit_id}" if config.SITE_URL else None
    alert.send(topic, reason, f"Visit #{visit_id} at the birdbath", photo, click)


class Watcher:
    def __init__(self):
        z = zones.current()
        self.detector = MotionDetector(z.roi, config.MOTION_MIN_AREA, z.ring, z.spots, z.ground)
        self.recalibrator = autocal.Recalibrator()
        self.busy = threading.Event()
        self.cooldown_until = 0.0  # no captures at all before this
        self.backoff_until = 0.0   # during rain backoff, captures are probes (local animal check first)
        self.false_streak = 0
        # The ground zone backs off on its own, so wind in the leaves never blinds the birdbath.
        self.ground_until = 0.0
        self.ground_backoff_until = 0.0
        self.ground_streak = 0
        self.budget_warned: dict[bool, date] = {}
        self.streak = 0  # consecutive substream frames of rim motion (TRIGGER_FRAMES of them is a trigger)
        # One visit per stay: the scene just before the last trigger, and the open stay after a real visit.
        self.empty: np.ndarray | None = None
        self.scene: Scene | None = None  # the same, with its masks, for counting the animals in the photo
        self.stay_pending: tuple[int, np.ndarray] | None = None  # set by _work, opened by the frame loop
        self.stay: stay.Stay | None = None

    def _work(self, motion: float, probe: bool, trigger: str) -> None:
        """The capture thread. Whatever happens, `busy` is cleared at the end: left set, the frame loop would
        skip every frame for good while the process stayed up (so systemd never restarted it)."""
        global last_visit_id
        visitor, empty, scene, last_visit_id = None, self.empty, self.scene, None
        try:
            try:
                visitor = handle_event(motion, probe=probe, trigger=trigger, scene=scene)
            except Exception:
                log.exception("failed to handle motion event")
            finally:
                self._settle(visitor, trigger, empty)
        except Exception:
            log.exception("failed to set the cooldown after a motion event")
        finally:
            self.busy.clear()

    def _settle(self, visitor: str | None, trigger: str, empty: np.ndarray | None) -> None:
        """Open the stay and set the cooldowns from what the capture found."""
        now = time.monotonic()
        if (config.STAY and visitor not in (None, "none", PROBE_EMPTY, PROBE_UNCHECKED, UNANSWERED) and last_visit_id
                and empty is not None):
            self.stay_pending = (last_visit_id, empty)
        # A probe nothing could filter counts as a false trigger, but the wait after it is the plain doubling one:
        # probing on every COOLDOWN_NONE without a verdict would just be asking Claude every 20 s.
        unchecked = visitor == PROBE_UNCHECKED
        if trigger == "ground":
            if visitor == PROBE_EMPTY:
                self.ground_until = now + config.COOLDOWN_NONE
            elif visitor == UNANSWERED:
                self.ground_until = now + config.COOLDOWN_VISIT  # neither a visit nor a false trigger
            else:
                cooldown, self.ground_streak = next_cooldown("none" if unchecked else visitor, self.ground_streak)
                if self.ground_streak:
                    if self.ground_streak > 1:
                        log.info("%d empty ground triggers in a row; ground backs off", self.ground_streak)
                    self.ground_until, self.ground_backoff_until = after_false(
                        now, self.ground_streak, cooldown, not unchecked and probe_ready())
                else:
                    self.ground_until, self.ground_backoff_until = now + cooldown, 0.0
        elif visitor == PROBE_EMPTY:
            self.cooldown_until = now + config.COOLDOWN_NONE  # keep the backoff; probe again later
        elif visitor == UNANSWERED:
            # Claude couldn't answer: the photo is kept for review. It says nothing about rain, so the backoff
            # streak is left as it was, and the usual wait after a visit keeps an outage from filling the queue.
            self.cooldown_until = now + config.COOLDOWN_VISIT
        else:
            cooldown, self.false_streak = next_cooldown("none" if unchecked else visitor, self.false_streak)
            if self.false_streak:
                if self.false_streak > 1:
                    log.info("%d false triggers in a row; backing off", self.false_streak)
                # With a trained animal check, keep probing every COOLDOWN_NONE during the backoff instead of
                # going blind.
                self.cooldown_until, self.backoff_until = after_false(
                    now, self.false_streak, cooldown, not unchecked and probe_ready())
            else:
                self.backoff_until = 0.0
                self.cooldown_until = now + cooldown

    def _maybe_trigger(self, motion: float, probe: bool = False, trigger: str = "rim") -> None:
        ground = trigger == "ground"
        if calls_today(ground) >= daily_limit(ground):
            if self.budget_warned.get(ground) != date.today():
                zone = "ground" if ground else "birdbath"
                log.warning("daily limit of %d %s visits reached; ignoring %s motion until tomorrow",
                            daily_limit(ground), zone, zone)
                self.budget_warned[ground] = date.today()
            if ground:
                self.ground_until = time.monotonic() + 300
            else:
                self.cooldown_until = time.monotonic() + 300
            return
        self.empty = self.detector.empty_scene()
        self.scene = self.detector.scene()
        self.busy.set()
        threading.Thread(target=self._work, args=(motion, probe, trigger), daemon=True).start()

    def follow_stay(self, now: float) -> bool:
        """Keep the last real visit open while its visitor is in view. True while a stay is open (no triggers).
        `now` is time.monotonic(); the stored end time comes from the wall clock, minus how long ago it was."""
        d = self.detector
        if self.stay is None and self.stay_pending is not None and d.full is not None:
            visit_id, empty = self.stay_pending
            self.stay_pending = None
            if empty.shape == d.full.shape:
                self.stay = stay.Stay(visit_id, empty, d.stay_mask(d.full.shape), d.night, now)
        if self.stay is None or d.full is None:
            return False
        st = self.stay
        reason = st.update(d.full, cv2.threshold(d.full_diff, 25, 255, cv2.THRESH_BINARY)[1], d.night, now)
        if reason is None:
            return True
        self.stay = None
        ended_at = st.moved if reason == stay.GONE else now
        ended = (datetime.now() - timedelta(seconds=now - ended_at)).replace(microsecond=0)
        try:
            with db.session() as conn:
                db.end_stay(conn, st.visit_id, ended.isoformat(), reason)
        except Exception:
            log.exception("could not save the end of visit %d", st.visit_id)
        log.info("visit %d ended (%s) after %d s", st.visit_id, reason, int(ended_at - st.started))
        self.cooldown_until = max(self.cooldown_until, now + config.STAY_REARM)
        self.ground_until = max(self.ground_until, now + config.STAY_REARM)
        self.detector.reset()  # start the next background from the empty scene
        return False

    def run(self) -> None:
        while True:
            cap = open_capture(config.rtsp_url(config.RTSP_SUB))
            if not cap.isOpened():
                log.warning("substream unavailable; retrying in 10 s")
                time.sleep(10)
                continue
            log.info("watching %s", redact(config.rtsp_url(config.RTSP_SUB)))
            self.detector.reset()
            self.streak = 0
            frames, last_frame = 0, time.monotonic()
            # Reconnect once no frame has arrived for STALL_SECONDS. Counting failed reads was no good: with
            # the FFmpeg backend each read of a stalled stream blocked for ~30 s, so 50 of them was 25 min blind.
            while time.monotonic() - last_frame < STALL_SECONDS:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue
                last_frame = time.monotonic()
                frames += 1
                try:
                    self.step(frame, frames)
                except Exception:
                    # A locked database (zones, the daily count, the recalibrator's log) must not take the process
                    # down with the rain backoff and the open stay in it; skip this frame and carry on.
                    log_rarely("frame", "couldn't process a frame; carrying on")
            log.warning("lost the substream; reconnecting")
            cap.release()
            time.sleep(2)

    def step(self, frame: np.ndarray, frames: int) -> None:
        """One substream frame: score the motion and, if it adds up to a trigger, start a capture."""
        if self.detector.use(zones.current()):
            log.info("zones recalibrated; switched to the new ones")
        motion = self.detector.score(frame)
        water, ground = self.detector.water_hit, self.detector.ground_hit
        quiet = frames > WARMUP_FRAMES and not self.busy.is_set() and motion < config.MOTION_MIN_AREA \
            and not water and not ground and self.stay is None
        self.recalibrator.feed(frame, self.detector.night, quiet)
        if frames <= WARMUP_FRAMES or self.busy.is_set():
            self.streak = 0
            return
        now = time.monotonic()
        if self.follow_stay(now):
            self.streak = 0
            return
        action = motion_action(now, self.cooldown_until, self.backoff_until, probe_ready())
        self.streak = self.streak + 1 if action != "wait" and motion >= config.MOTION_MIN_AREA else 0
        if action != "wait" and (self.streak >= config.TRIGGER_FRAMES or water):
            self.streak = 0
            trigger = "rim" if streak_hit(motion) else "water"
            # Water-only triggers always go through the local animal check first, so a ripple that slips
            # past the blob rule costs a few seconds of Pi CPU, never a Claude call.
            probe = action == "probe" or (trigger == "water" and probe_ready())
            log.info("motion %.3f (%s%s)", motion, trigger, ", probe" if probe else "")
            self._maybe_trigger(motion, probe=probe, trigger=trigger)
            if self.detector.water:
                self.detector.water.reset()
        elif ground and (ground_action := motion_action(now, self.ground_until, self.ground_backoff_until,
                                                        probe_ready())) != "wait":
            # Gated only by the ground's own cooldown, backoff and budget (plus `busy` and an open stay, above):
            # the bath resting after a rainy false alarm, or over its daily limit, must not hold a cat back.
            # Claude, under the ground's own limit, except during a run of empty ground triggers: then
            # probes, which pass unless the animal check calls them empty. Empty answers are kept as
            # false triggers.
            probe = ground_action == "probe"
            log.info("motion on the ground (%s%s)", "night" if self.detector.night else "day",
                     ", probe" if probe else "")
            self._maybe_trigger(motion, probe=probe, trigger="ground")
            if self.detector.ground:
                self.detector.ground.reset()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="capture and identify one frame now, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    with db.session() as conn:
        db.init_db(conn)
    if args.once:
        handle_event()
        return
    weather.start_background()  # hourly weather in its own thread and DB connection, never on the frame loop
    watcher = Watcher()
    # The animal check retrains in its own thread, never on a trigger's path; it also looks through the bursts of
    # corrected false triggers, pausing while a capture runs.
    presence.start_background(busy=watcher.busy)
    # The photo budget: thins older visits once a day and frees space when over the budget, in a low-priority thread
    # that pauses while a capture runs.
    retention.start_background(busy=watcher.busy)
    watcher.run()


if __name__ == "__main__":
    main()
