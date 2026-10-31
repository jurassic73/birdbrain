"""Watch the camera for motion and record visits.

python -m birdbrain.watch          run forever
python -m birdbrain.watch --once   capture and identify one frame now, then exit
"""

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

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402  (must come after the capture options are set)
import numpy as np  # noqa: E402

from . import alert, autocal, classifier, config, db, detector, identify, images, presence, retention, scenery, species_info, stay, zones  # noqa: E402

log = logging.getLogger("birdbrain.watch")

WARMUP_FRAMES = 30  # let the background model settle after (re)connecting
NIGHT_CHECK_FRAMES = 15  # how often (substream frames, ~1 s) to re-check whether the camera is in IR mode
last_visit_id: int | None = None  # set by handle_event, so the watcher can keep that visit open while it stays


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
    """Visits today that Claude was asked about (visits the local detector settled cost nothing), counted
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


_mode_cache: tuple[float, str, str] = (0.0, "", "")  # (read at, .env default, mode)


def after_false(now: float, streak: int, cooldown: int, can_probe: bool) -> tuple[float, float]:
    """(cooldown_until, backoff_until) after a false trigger, the streak-th in a row. During the backoff,
    triggers are probes every COOLDOWN_NONE (with a local detector). From PROBE_AFTER in a row the backoff
    runs FULL_CHECK_EVERY, so steady rain or blowing leaves cost one Claude call per half hour, not one
    every COOLDOWN_NONE_MAX (about 240 a night on 2026-10-26)."""
    if not can_probe:
        return now + cooldown, now + cooldown
    backoff = config.FULL_CHECK_EVERY if streak >= config.PROBE_AFTER else cooldown
    return now + config.COOLDOWN_NONE, now + backoff


def probe_passes(det: dict | None) -> bool:
    """A probe goes on to Claude when the detector sees an animal or the scenery check finds something new
    (it scores squirrels, which COCO has no class for)."""
    if det is None:
        return False
    return bool(det["animal"]) or ("scenery" in det and det["scenery"] is not None and not det["scenery"]["empty"])


def detector_mode() -> str:
    """The detector mode from Settings (else .env), re-read at most every 15 s since motion checks it per frame."""
    global _mode_cache
    read_at, default, mode = _mode_cache
    if time.monotonic() - read_at > 15 or default != config.DETECTOR:
        with db.session() as conn:
            mode = db.detector_mode(conn)
        _mode_cache = (time.monotonic(), config.DETECTOR, mode)
    return mode


def should_skip_claude(det: dict | None, night: bool) -> bool:
    """Gate mode only: skip Claude when the local detector saw nothing. Never at night, when raccoons and
    possums (which the detector doesn't know) come by."""
    return detector_mode() == "gate" and det is not None and not det["animal"] and not night


PROBE_EMPTY = "probe-empty"  # a backoff probe the local detector found nothing in: not stored, no Claude call


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


def detector_ready() -> bool:
    return detector_mode() != "off" and config.DETECTOR_MODEL.exists()


def motion_action(now: float, cooldown_until: float, backoff_until: float, can_probe: bool) -> str:
    """What motion should do right now: "wait", "probe" (during rain backoff: local detector first, Claude
    only if it sees an animal) or "full" (normal capture and identify)."""
    if now < cooldown_until:
        return "wait"
    if now < backoff_until:
        return "probe" if can_probe else "wait"
    return "full"


def handle_event(motion: float | None = None, probe: bool = False, trigger: str = "rim") -> str | None:
    """Capture, identify and store one visit. Returns the visitor type, None if nothing was captured, or
    PROBE_EMPTY for a backoff probe where the local detector saw nothing (discarded: rain shouldn't fill
    the disk or the bill)."""
    captured_at = datetime.now().replace(microsecond=0)
    frames = capture_burst()
    if not frames:
        return None
    z = zones.current()
    crop = z.ground_crop if trigger == "ground" else z.crop
    ranked = best_frames(frames, crop)
    best = ranked[0]
    night = is_night(best)
    det = None
    try:
        det = detector.run(best, crop)
        if det is not None:
            with db.session() as conn:
                det["scenery"] = scenery.check(conn, det, trigger == "ground", night)
    except Exception:
        log.exception("local detector failed; asking Claude anyway")
    if probe and not probe_passes(det):
        log.info("probe (%s trigger): detector saw nothing; discarded", trigger)
        with db.session() as conn:
            db.bump(conn, "probe_empty")
        return PROBE_EMPTY
    # The local animal check: BioCLIP 2's embedding of the crop box, scored by a model trained on this yard's
    # labelled frames. In gate mode an empty verdict skips Claude (day and night; the owner reviews them).
    emb = check = None
    pmode = presence_mode()
    if pmode != "off":
        emb = classifier.box_embedding(best, crop)
        try:
            with db.session() as conn:
                check = presence.check(conn, emb)
        except Exception:
            log.exception("local animal check failed; asking Claude")
    if pmode == "gate" and check and check.get("empty"):
        result = identify.settled_by_presence(check, det)
    elif should_skip_claude(det, night):
        result = identify.settled_locally(det)
        if check:
            result.raw["presence"] = check
    else:
        with db.session() as conn:
            lessons = db.recent_lessons(conn)
            site = db.prompt_site(conn)
            confirmed = db.confirmed_species(conn)
        result = identify.identify(best, captured_at, crop, lessons, extra_frames=ranked[1:], site=site,
                                   confirmed=confirmed)
        if det is not None:
            result.raw["detector"] = det
        if check:
            result.raw["presence"] = check
    # Shadow only: the local species classifier records its guess for the Stats scorecard; Claude's label stands.
    if result.sighting.visitor != "none" and (guess := classifier.run(best, result.raw.get("circle"), crop, night)):
        result.raw["classifier"] = guess
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
            images.store_burst(frames, image_rel, max_edge=max_edge, quality=quality)
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
    for _, label in seen:
        species_info.prefetch(label)  # no-op once the card is cached
    det_note = "" if det is None else f", detector: {det['best']['label'] if det['best'] else 'nothing'} {det['seconds']}s"
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
        self.backoff_until = 0.0   # during rain backoff, captures are probes (local detector first)
        self.false_streak = 0
        # The ground zone backs off on its own, so wind in the leaves never blinds the birdbath.
        self.ground_until = 0.0
        self.ground_backoff_until = 0.0
        self.ground_streak = 0
        self.budget_warned: dict[bool, date] = {}
        # One visit per stay: the scene just before the last trigger, and the open stay after a real visit.
        self.empty: np.ndarray | None = None
        self.stay_pending: tuple[int, np.ndarray] | None = None  # set by _work, opened by the frame loop
        self.stay: stay.Stay | None = None

    def _work(self, motion: float, probe: bool, trigger: str) -> None:
        global last_visit_id
        visitor, empty, last_visit_id = None, self.empty, None
        try:
            visitor = handle_event(motion, probe=probe, trigger=trigger)
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
                            now, self.ground_streak, cooldown, detector_ready())
                    else:
                        self.ground_until, self.ground_backoff_until = now + cooldown, 0.0
            elif visitor == PROBE_EMPTY:
                self.cooldown_until = now + config.COOLDOWN_NONE  # keep the backoff; probe again later
            else:
                cooldown, self.false_streak = next_cooldown(visitor, self.false_streak)
                if self.false_streak:
                    if self.false_streak > 1:
                        log.info("%d false triggers in a row; backing off", self.false_streak)
                    # With a detector, keep probing every COOLDOWN_NONE during the backoff instead of going blind.
                    self.cooldown_until, self.backoff_until = after_false(
                        now, self.false_streak, cooldown, detector_ready())
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
                action = motion_action(now, self.cooldown_until, self.backoff_until, detector_ready())
                streak = streak + 1 if action != "wait" and motion >= config.MOTION_MIN_AREA else 0
                if action != "wait" and (streak >= config.TRIGGER_FRAMES or water):
                    streak = 0
                    trigger = "rim" if streak_hit(motion) else "water"
                    # Water-only triggers always go through the local detector first, so a ripple that slips
                    # past the blob rule costs half a second of CPU, never a Claude call.
                    probe = action == "probe" or (trigger == "water" and detector_ready())
                    log.info("motion %.3f (%s%s)", motion, trigger, ", probe" if probe else "")
                    self._maybe_trigger(motion, probe=probe, trigger=trigger)
                    if self.detector.water:
                        self.detector.water.reset()
                elif ground and action != "wait" and (
                        ground_action := motion_action(now, self.ground_until, self.ground_backoff_until,
                                                       detector_ready())) != "wait":
                    # Claude, under the ground's own limit, except during a run of empty ground triggers: then
                    # probes, which pass on anything the scenery check finds (the class detector alone threw
                    # away a real squirrel on 2026-10-26). Empty answers are kept as false triggers.
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
    Watcher().run()


if __name__ == "__main__":
    main()
