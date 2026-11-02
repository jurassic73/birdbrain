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
from datetime import date, datetime
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
last_visit_id: int | None = None  # set by handle_event, so the watcher can keep that visit open while it stays



_push: subprocess.Popen | None = None


def push_public() -> None:
    """Send the new visit to the public mirror now instead of waiting for cron (PUBLIC_DEST in .env).
    Runs in the background; skipped if the previous push is still going (the next one catches up)."""
    global _push
    if not os.environ.get("PUBLIC_DEST") or (_push is not None and _push.poll() is None):
        return
    script = Path(__file__).resolve().parent.parent / "scripts" / "push_public.sh"
    if script.exists():
        _push = subprocess.Popen([str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def redact(text: str) -> str:
    return re.sub(r"rtsp://[^@/\s]*@", "rtsp://***@", text)


def roi(frame: np.ndarray, box) -> np.ndarray:
    if not box:
        return frame
    H, W = frame.shape[:2]
    x, y, w, h = box
    return frame[int(y * H):int((y + h) * H), int(x * W):int((x + w) * W)]


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
        gray = roi(full, self.box)
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
        diff = roi(full_diff, self.box)
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


    def stay_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Everywhere a visitor can be, in full-frame coordinates: rim, rocks, water and ground."""
        if self.ring:
            mask = motion_mask(shape, self.ring, self.spots) | water_mask(shape, self.ring, self.spots)
        else:
            mask = np.zeros(shape, np.uint8)
            h, w = shape
            x, y, bw, bh = self.box or (0, 0, 1, 1)
            mask[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)] = 255
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
    gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(roi(gray, crop), cv2.CV_64F).var())


def best_frames(frames: list[bytes], crop=None, n: int = 3) -> list[bytes]:
    """The burst's sharpest frames inside the birdbath, best first."""
    return sorted(frames, key=lambda f: sharpness(f, crop), reverse=True)[:n]


def frame_is_night(bgr: np.ndarray) -> bool:
    """IR night frames are effectively grayscale, so color saturation is near zero."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < 12


def is_night(jpeg: bytes) -> bool:
    return frame_is_night(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))


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
    with db.session() as conn:
        return conn.execute(
            """SELECT COUNT(*) FROM visits WHERE date(captured_at) = ? AND model LIKE 'claude%'
                 AND (COALESCE(json_extract(raw_json, '$.trigger'), '') = 'ground') = ?""",
            (date.today().isoformat(), int(ground)),
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
    every COOLDOWN_NONE_MAX (about 240 a night on 2026-10-26)."""
    if not can_probe:
        return now + cooldown, now + cooldown
    backoff = config.FULL_CHECK_EVERY if streak >= config.PROBE_AFTER else cooldown
    return now + config.COOLDOWN_NONE, now + backoff


def probe_passes(check: dict | None) -> bool:
    """A rain-backoff probe goes on to Claude unless the local animal check calls it empty. With the check
    off, or before it has enough labelled frames to train, every probe goes to Claude."""
    return not (check and check.get("empty"))


PROBE_EMPTY = "probe-empty"  # a backoff probe the local animal check found empty in: not stored, no Claude call


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


def probe_ready() -> bool:
    """Probes need the local animal check: without it a rain run just waits out its backoff."""
    return presence_mode() != "off"


def motion_action(now: float, cooldown_until: float, backoff_until: float, can_probe: bool) -> str:
    """What motion should do right now: "wait", "probe" (during rain backoff: local animal check first, Claude
    only if it sees an animal) or "full" (normal capture and identify)."""
    if now < cooldown_until:
        return "wait"
    if now < backoff_until:
        return "probe" if can_probe else "wait"
    return "full"


def _scored(jpeg: bytes, crop) -> tuple[np.ndarray | None, dict | None]:
    emb = classifier.box_embedding(jpeg, crop)
    with db.session() as conn:
        return emb, presence.check(conn, emb)


def animal_check(frames: list[bytes], ranked: list[bytes], crop) -> tuple[np.ndarray | None, dict | None, list[bytes]]:
    """(embedding, verdict, ranked frames) from the local animal check, fail-soft (no verdict on any problem).

    It scores the sharpest frame and, when that looks empty and PRESENCE_LAST_FRAME is on, the burst's last frame
    too: a landing bird is usually in the bowl by then, while sharpness can favour ripples or mulch. The higher
    score counts (so the verdict is the max of the two), and its embedding is the one stored for training. When
    only the last frame shows an animal it becomes the photo, so Claude sees what the Pi saw."""
    try:
        emb, check = _scored(ranked[0], crop)
    except Exception:
        log.exception("local animal check failed; asking Claude")
        return None, None, ranked
    last = frames[-1]
    if (not config.PRESENCE_LAST_FRAME or not check or check.get("score") is None or not check.get("empty")
            or last is ranked[0]):
        return emb, check, ranked
    try:
        emb2, check2 = _scored(last, crop)
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
    """Capture, identify and store one visit. Returns the visitor type, None if nothing was captured, or
    PROBE_EMPTY for a backoff probe where the local animal check found nothing (discarded: rain shouldn't fill
    the disk or the bill). `scene` is the empty scene from the trigger, used to count the animals in view."""
    captured_at = datetime.now().replace(microsecond=0)
    frames = capture_burst()
    if not frames:
        return None
    z = zones.current()
    crop = z.ground_crop if trigger == "ground" else z.crop
    ranked = best_frames(frames, crop)
    # The local animal check: BioCLIP 2's embedding of the crop box, scored by a model trained on this yard's
    # labelled frames. It is the only gate: in gate mode an empty verdict skips Claude (day and night; the
    # owner reviews them), and a rain-backoff probe it calls empty is discarded in any mode but off.
    emb = check = None
    pmode = presence_mode()
    if pmode != "off":
        emb, check, ranked = animal_check(frames, ranked, crop)
    best = ranked[0]
    night = is_night(best)
    if probe and not probe_passes(check):
        log.info("probe (%s trigger): animal check found nothing; discarded", trigger)
        with db.session() as conn:
            db.bump(conn, "probe_empty")
        return PROBE_EMPTY
    empty = bool(check and check.get("empty"))
    guess = pick = None
    animals = None if empty else count_animals(scene, best, crop)
    crowd = None
    if not empty:  # the species gate: a sure daytime name of a species already settled here skips Claude
        with db.session() as conn:
            settled = db.settled_species(conn)
        guess, pick = classifier.gate_check(emb, night, settled)
        if pick and animals and animals > 1:
            # One BioCLIP guess names one animal; Claude counts them and names any other kind.
            log.info("species gate: %d animals in view, asking Claude instead of naming %s", animals, pick[0])
            crowd, pick = pick[0], None
    if pick:
        result = identify.settled_by_classifier(guess, pick, check)
        if config.CIRCLE_FROM_MOTION:
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
        local_check = None if night else (lambda circle: classifier.run(best, circle, crop, night))
        result = identify.identify(best, captured_at, crop, lessons, extra_frames=ranked[1:], site=site,
                                   confirmed=confirmed, local_check=local_check)
        if check:
            result.raw["presence"] = check
    # When Claude answered, the classifier records its guess from Claude's circle for the Stats scorecard (already
    # made on Haiku's circle when it decided whether to ask Sonnet; again on Sonnet's circle if Sonnet answered).
    if (not pick and result.sighting.visitor != "none" and (result.escalated or "classifier" not in result.raw)
            and (guess := classifier.run(best, result.raw.get("circle"), crop, night))):
        result.raw["classifier"] = guess
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
    # Every kind of animal in the photo: the main one first, then the others Claude listed.
    seen = [] if false_trigger else [(s.visitor, s.species or s.visitor.capitalize()),
                                     *((a["visitor"], db.also_label(a)) for a in also)]
    global last_visit_id
    with db.session() as conn:
        # Check BEFORE inserting so the new-species count doesn't include this visit yet.
        for visitor, label in seen:
            try:
                alert_reason = alert.reason(conn, visitor, label, captured_at)
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
    last_visit_id = visit_id
    if false_trigger:
        with db.session() as conn:
            if pruned := retention.prune(conn):
                log.info("false-trigger storage limit: deleted the %d oldest", pruned)
    if alert_reason:
        with db.session() as conn:
            topic = db.get_value(conn, "ntfy_topic") or ""
        click = f"{config.SITE_URL}/visit/{visit_id}" if config.SITE_URL else None
        alert.send(topic, alert_reason, f"Visit #{visit_id} at the birdbath", best, click)
    if not false_trigger:
        push_public()
    for _, label in seen:
        species_info.prefetch(label)  # no-op once the card is cached
    det_note = "" if not check or check.get("score") is None else f", animal check {check['score']:.2f}"
    log.info("visit %d: %s %s (%s, %s%s%s)", visit_id, s.visitor, s.species or "", s.confidence,
             result.model, ", escalated" if result.escalated else "", det_note)
    return s.visitor


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
        # One visit per stay: the scene just before the last trigger, and the open stay after a real visit.
        self.empty: np.ndarray | None = None
        self.scene: Scene | None = None  # the same, with its masks, for counting the animals in the photo
        self.stay_pending: tuple[int, np.ndarray] | None = None  # set by _work, opened by the frame loop
        self.stay: stay.Stay | None = None

    def _work(self, motion: float, probe: bool, trigger: str) -> None:
        global last_visit_id
        visitor, empty, scene, last_visit_id = None, self.empty, self.scene, None
        try:
            visitor = handle_event(motion, probe=probe, trigger=trigger, scene=scene)
        except Exception:
            log.exception("failed to handle motion event")
        finally:
            now = time.time()
            if config.STAY and visitor not in (None, "none", PROBE_EMPTY) and last_visit_id and empty is not None:
                self.stay_pending = (last_visit_id, empty)
            if trigger == "ground":
                if visitor == PROBE_EMPTY:
                    self.ground_until = now + config.COOLDOWN_NONE
                else:
                    cooldown, self.ground_streak = next_cooldown(visitor, self.ground_streak)
                    if self.ground_streak:
                        if self.ground_streak > 1:
                            log.info("%d empty ground triggers in a row; ground backs off", self.ground_streak)
                        self.ground_until, self.ground_backoff_until = after_false(
                            now, self.ground_streak, cooldown, probe_ready())
                    else:
                        self.ground_until, self.ground_backoff_until = now + cooldown, 0.0
            elif visitor == PROBE_EMPTY:
                self.cooldown_until = now + config.COOLDOWN_NONE  # keep the backoff; probe again later
            else:
                cooldown, self.false_streak = next_cooldown(visitor, self.false_streak)
                if self.false_streak:
                    if self.false_streak > 1:
                        log.info("%d false triggers in a row; backing off", self.false_streak)
                    # With the animal check, keep probing every COOLDOWN_NONE during the backoff instead of going blind.
                    self.cooldown_until, self.backoff_until = after_false(
                        now, self.false_streak, cooldown, probe_ready())
                else:
                    self.backoff_until = 0.0
                    self.cooldown_until = now + cooldown
            self.busy.clear()

    def _maybe_trigger(self, motion: float, probe: bool = False, trigger: str = "rim") -> None:
        ground = trigger == "ground"
        if calls_today(ground) >= daily_limit(ground):
            if self.budget_warned.get(ground) != date.today():
                zone = "ground" if ground else "birdbath"
                log.warning("daily limit of %d %s visits reached; ignoring %s motion until tomorrow",
                            daily_limit(ground), zone, zone)
                self.budget_warned[ground] = date.today()
            if ground:
                self.ground_until = time.time() + 300
            else:
                self.cooldown_until = time.time() + 300
            return
        self.empty = self.detector.empty_scene()
        self.scene = self.detector.scene()
        self.busy.set()
        threading.Thread(target=self._work, args=(motion, probe, trigger), daemon=True).start()

    def follow_stay(self, now: float) -> bool:
        """Keep the last real visit open while its visitor is in view. True while a stay is open (no triggers)."""
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
        ended = datetime.fromtimestamp(st.moved if reason == stay.GONE else now).replace(microsecond=0)
        try:
            with db.session() as conn:
                db.end_stay(conn, st.visit_id, ended.isoformat(), reason)
        except Exception:
            log.exception("could not save the end of visit %d", st.visit_id)
        log.info("visit %d ended (%s) after %d s", st.visit_id, reason, int(ended.timestamp() - st.started))
        self.cooldown_until = max(self.cooldown_until, now + config.STAY_REARM)
        self.ground_until = max(self.ground_until, now + config.STAY_REARM)
        self.detector.reset()  # start the next background from the empty scene
        return False

    def run(self) -> None:
        while True:
            cap = cv2.VideoCapture(config.rtsp_url(config.RTSP_SUB), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log.warning("substream unavailable; retrying in 10 s")
                time.sleep(10)
                continue
            log.info("watching %s", redact(config.rtsp_url(config.RTSP_SUB)))
            self.detector.reset()
            frames, streak, failures = 0, 0, 0
            while failures < 50:
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    time.sleep(0.1)
                    continue
                failures = 0
                frames += 1
                if self.detector.use(zones.current()):
                    log.info("zones recalibrated; switched to the new ones")
                motion = self.detector.score(frame)
                water, ground = self.detector.water_hit, self.detector.ground_hit
                quiet = frames > WARMUP_FRAMES and not self.busy.is_set() and motion < config.MOTION_MIN_AREA \
                    and not water and not ground and self.stay is None
                self.recalibrator.feed(frame, self.detector.night, quiet)
                if frames <= WARMUP_FRAMES or self.busy.is_set():
                    streak = 0
                    continue
                now = time.time()
                if self.follow_stay(now):
                    streak = 0
                    continue
                action = motion_action(now, self.cooldown_until, self.backoff_until, probe_ready())
                streak = streak + 1 if action != "wait" and motion >= config.MOTION_MIN_AREA else 0
                if action != "wait" and (streak >= config.TRIGGER_FRAMES or water):
                    streak = 0
                    trigger = "rim" if streak_hit(motion) else "water"
                    # Water-only triggers always go through the local animal check first, so a ripple that slips
                    # past the blob rule costs a few seconds of Pi CPU, never a Claude call.
                    probe = action == "probe" or (trigger == "water" and probe_ready())
                    log.info("motion %.3f (%s%s)", motion, trigger, ", probe" if probe else "")
                    self._maybe_trigger(motion, probe=probe, trigger=trigger)
                    if self.detector.water:
                        self.detector.water.reset()
                elif ground and action != "wait" and (
                        ground_action := motion_action(now, self.ground_until, self.ground_backoff_until,
                                                       probe_ready())) != "wait":
                    # Claude, under the ground's own limit, except during a run of empty ground triggers: then
                    # probes, which pass unless the animal check calls them empty. Empty answers are kept as
                    # false triggers.
                    probe = ground_action == "probe"
                    log.info("motion on the ground (%s%s)", "night" if self.detector.night else "day",
                             ", probe" if probe else "")
                    self._maybe_trigger(motion, probe=probe, trigger="ground")
                    self.detector.ground.reset()
            log.warning("lost the substream; reconnecting")
            cap.release()
            time.sleep(2)


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
    watcher.run()


if __name__ == "__main__":
    main()
