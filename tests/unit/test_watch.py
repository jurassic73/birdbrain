"""Unit tests of birdbrain/watch: pure functions, no files, database or network."""

import sqlite3

import cv2
import numpy as np
import pytest

from helpers.watch import (GROUND, GROUND_Y, H, RING, SPOTS, W, WATER_SPOT, frame, ground_detector, ground_hits,
                           locked, night_frame, run_frames, walker, water_detector, with_blob)
from birdbrain import clips, config, stay, watch


@pytest.mark.parametrize(
    "visitor, streak, expected",
    [
        ("bird", 5, (60, 0)),      # a real visitor resets the backoff
        ("none", 0, (20, 1)),
        ("none", 1, (40, 2)),
        ("none", 3, (160, 4)),
        ("none", 9, (300, 10)),    # capped
        (None, 0, (20, 1)),        # nothing captured counts as a false trigger
    ],
)
def test_false_trigger_backoff(monkeypatch, visitor, streak, expected):
    monkeypatch.setattr(config, "COOLDOWN_VISIT", 60)
    monkeypatch.setattr(config, "COOLDOWN_NONE", 20)
    monkeypatch.setattr(config, "COOLDOWN_NONE_MAX", 300)
    assert watch.next_cooldown(visitor, streak) == expected


def jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_motion_mask_is_the_rim_plus_spots_not_the_water():
    mask = watch.motion_mask((H, W), RING, SPOTS)
    cx, cy = W // 2, H // 2
    assert mask[cy, cx] == 0                          # open water
    assert mask[cy, int(0.5 * W + 0.175 * W)] == 255  # on the rim, between inner and outer ellipse
    assert mask[cy, int(0.42 * W)] == 255             # the rock spot inside the water
    assert mask[5, 5] == 0                            # outside the bath


def test_motion_mask_inner_center_offset_moves_the_water():
    shifted = watch.motion_mask((H, W), RING + (0.45,))
    # With the water ellipse raised, a point just below its new bottom edge is rim.
    y = int(0.45 * H + 0.18 * H) + 3
    assert shifted[y, W // 2] == 255
    assert watch.motion_mask((H, W), RING)[y, W // 2] == 0


def score_after(detector, moving: np.ndarray, still_frames: int = 5) -> float:
    for _ in range(still_frames):
        detector.score(frame())
    return detector.score(moving)


def test_still_scene_scores_zero():
    d = watch.MotionDetector(None, 0.02, RING, SPOTS)
    assert all(d.score(frame()) == 0 for _ in range(10))


def test_bird_on_the_rim_triggers_but_ripples_in_the_water_do_not():
    rim_x = int(0.5 * W + 0.175 * W)
    on_rim = score_after(watch.MotionDetector(None, 0.02, RING, SPOTS), with_blob(rim_x, H // 2))
    in_water = score_after(watch.MotionDetector(None, 0.02, RING, SPOTS), with_blob(W // 2 + 20, H // 2 + 10, r=10))
    on_rock = score_after(watch.MotionDetector(None, 0.02, RING, SPOTS), with_blob(int(0.42 * W), H // 2, r=10))
    assert on_rim > 0.02
    assert in_water == 0
    assert on_rock > 0.02


def test_rectangular_roi_without_a_ring():
    d = watch.MotionDetector((0.0, 0.0, 0.5, 0.5), 0.004)
    assert score_after(d, with_blob(int(0.75 * W), int(0.75 * H))) == 0   # outside the ROI
    d.reset()
    assert score_after(d, with_blob(int(0.25 * W), int(0.25 * H))) > 0.004


def test_water_mask_is_the_inner_ellipse_minus_rocks():
    mask = watch.water_mask((H, W), RING, SPOTS)
    assert mask[WATER_SPOT[1], WATER_SPOT[0]] == 255
    assert mask[H // 2, int(0.42 * W)] == 0                 # the rock spot belongs to the rim zone
    assert mask[H // 2, int(0.5 * W + 0.175 * W)] == 0      # the rim itself


def test_a_bird_sitting_in_the_water_triggers_after_it_persists():
    hits = run_frames(water_detector(), [with_blob(*WATER_SPOT, r=12)] * 12)
    assert hits[: 9] == [False] * 9 and hits[9] is True   # WATER_PERSIST_FRAMES = 10


def test_a_bird_moving_a_little_while_bathing_still_counts():
    path = [with_blob(WATER_SPOT[0] + 3 * i, WATER_SPOT[1], r=12) for i in range(12)]
    assert any(run_frames(water_detector(), path))


def test_a_one_frame_flicker_does_not_trigger():
    frames = ([with_blob(*WATER_SPOT, r=12)] + [frame()] * 3) * 4
    assert not any(run_frames(water_detector(), frames))


def test_the_whole_water_changing_at_once_is_not_a_bird():
    dark = frame()
    cv2.ellipse(dark, (W // 2, H // 2), (int(0.15 * W), int(0.18 * H)), 0, 0, 360, (20, 20, 20), -1)
    assert not any(run_frames(water_detector(), [dark] * 8))


def test_bird_in_water_does_not_touch_the_rim_score():
    d = water_detector()
    assert all(d.score(with_blob(*WATER_SPOT, r=12)) == 0 for _ in range(10))  # rim untouched...
    assert d.water_hit                                                        # ...but the water zone fired


@pytest.mark.parametrize(
    "now, cooldown, backoff, can_probe, expected",
    [
        (100, 110, 0, True, "wait"),       # inside the hard cooldown
        (100, 90, 200, True, "probe"),     # rain backoff with the animal check: probe
        (100, 90, 200, False, "wait"),     # rain backoff without it: stay quiet, like before
        (100, 90, 95, True, "full"),       # backoff over
    ],
)
def test_motion_action(now, cooldown, backoff, can_probe, expected):
    assert watch.motion_action(now, cooldown, backoff, can_probe) == expected


def test_sharpness_prefers_the_sharp_frame():
    sharp = frame()
    cv2.putText(sharp, "JUNCO", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    blurry = cv2.GaussianBlur(sharp, (31, 31), 0)
    assert watch.sharpness(jpeg(sharp)) > watch.sharpness(jpeg(blurry)) * 2


def test_best_frames_judges_sharpness_inside_the_crop_only():
    crop = (0.25, 0.25, 0.5, 0.5)
    rng = np.random.default_rng(0)
    mulch = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)            # fine texture everywhere...
    busy_outside = mulch.copy()
    busy_outside[H // 4:3 * H // 4, W // 4:3 * W // 4] = (90, 110, 80)  # ...except a smooth, empty birdbath
    bird_inside = busy_outside.copy()
    cv2.circle(bird_inside, (W // 2, H // 2), 30, (255, 255, 255), 3)  # a crisp bird in the bath
    bird_inside[:H // 4] = cv2.GaussianBlur(bird_inside[:H // 4], (9, 9), 0)  # background a bit softer
    plain = frame()
    ranked = watch.best_frames([jpeg(busy_outside), jpeg(plain), jpeg(bird_inside)], crop, n=2)
    assert ranked[0] == jpeg(bird_inside) and len(ranked) == 2
    # Judged on the whole frame, the busy background would have won.
    assert watch.sharpness(jpeg(busy_outside)) > watch.sharpness(jpeg(bird_inside))


def test_is_night_detects_grayscale_ir_frames():
    color = frame((40, 160, 60))
    gray = cv2.cvtColor(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert watch.is_night(jpeg(gray)) is True
    assert watch.is_night(jpeg(color)) is False


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Error opening rtsp://thingino:p%40ss@cam:554/ch0", "Error opening rtsp://***@cam:554/ch0"),
        ("no url here", "no url here"),
    ],
)
def test_redact_hides_credentials(text, expected):
    assert watch.redact(text) == expected


def test_ground_mask_leaves_the_bath_to_the_rim_and_water_zones():
    mask = watch.ground_mask((H, W), GROUND, RING)
    assert mask[GROUND_Y, 20] == 255                          # ground
    assert mask[int(0.7 * H), W // 2] == 0                    # the bath's lower rim, inside the zone's polygon
    assert mask[int(0.3 * H), 20] == 0                        # outside the polygon


def test_night_is_checked_from_the_substream():
    d = watch.MotionDetector(None, 0.02, RING, SPOTS, GROUND)
    d.score(night_frame())
    assert d.night is True
    d.reset()
    d.score(frame())
    assert d.night is False


def test_a_cat_walking_by_at_night_triggers_the_ground_zone():
    base = night_frame()
    hits = ground_hits(ground_detector(base), [walker(base, 100 + 4 * i, r=28) for i in range(12)])
    assert hits[:7] == [False] * 7 and any(hits[7:])          # GROUND_PERSIST_FRAMES = 8


def test_the_whole_ground_changing_at_once_is_light_not_an_animal():
    base = night_frame()
    lit = base.copy()
    lit[int(0.6 * H):] = (200, 200, 200)                          # headlights, or the dusk IR switch
    assert not any(ground_hits(ground_detector(base), [lit] * 12))


def test_long_runs_of_false_triggers_probe_for_half_an_hour():
    short = watch.after_false(1000.0, 1, 20, can_probe=True)
    long = watch.after_false(1000.0, watch.config.PROBE_AFTER, 300, can_probe=True)
    assert short == (1000.0 + watch.config.COOLDOWN_NONE, 1020.0)
    assert long == (1000.0 + watch.config.COOLDOWN_NONE, 1000.0 + watch.config.FULL_CHECK_EVERY)
    assert watch.after_false(1000.0, 9, 300, can_probe=False) == (1300.0, 1300.0)  # no animal check: just wait


def test_probe_passes_only_with_a_verdict_that_is_not_empty():
    assert watch.probe_passes({"empty": False, "score": 0.9})
    assert not watch.probe_passes({"empty": True, "score": 0.1})
    assert not watch.probe_passes(None)  # check off or failed: nothing filtered the probe, so it isn't one
    assert not watch.probe_passes({"empty": False, "score": None})  # not trained yet: the same


def test_classifier_usable_and_presence_trained_read_their_state(monkeypatch):
    monkeypatch.setattr(watch.config, "CLASSIFIER", "off")
    assert watch.classifier.usable() is False
    monkeypatch.setattr(watch.config, "CLASSIFIER", "gate")
    monkeypatch.setattr(watch.classifier, "_classifier", None)
    monkeypatch.setattr(watch.classifier, "_failed_at", 0.0)
    assert watch.classifier.usable() is True  # not tried yet: the first full check loads it
    monkeypatch.setattr(watch.classifier, "_failed_at", watch.time.monotonic())  # the 300 MB download failed
    assert watch.classifier.usable() is False
    monkeypatch.setattr(watch.classifier, "_classifier", object())
    assert watch.classifier.usable() is True
    assert watch.presence.trained() is False  # no model in this process yet
    monkeypatch.setattr(watch.presence, "_model", {"key": (), "model": None})  # trained, but on too few frames
    assert watch.presence.trained() is False
    monkeypatch.setattr(watch.presence, "_model", {"key": (), "model": {"threshold": 0.1}})
    assert watch.presence.trained() is True


def test_roi_outside_the_frame_is_none_not_an_empty_slice():
    img = frame()
    assert watch.roi(img, (-0.1, 0.5, 0.05, 0.3)) is None               # wholly outside: nothing, no wrapped slice
    assert watch.roi(img, (-0.05, -0.05, 0.3, 0.3)).shape[:2] == (90, 160)  # pushed over the corner: what's left
    assert watch.roi(img, None) is img
    assert watch.sharpness(jpeg(img), (-0.1, 0.5, 0.05, 0.3)) == 0.0   # used to raise in cv2 on every ground trigger
    assert watch.best_frames([jpeg(img)] * 2, (-0.1, 0.5, 0.05, 0.3), n=1) == [jpeg(img)]
    m = stay.box_mask((H, W), (-0.05, 0.9, 0.2, 0.2))
    assert m[H - 1, 0] == 255 and m[H - 1, W - 1] == 0 and m[0, 0] == 0 and cv2.countNonZero(m) == 96 * 36
    assert cv2.countNonZero(stay.box_mask((H, W), (1.2, 0.5, 0.2, 0.2))) == 0


def test_motion_detector_with_a_box_over_the_edge_still_scores():
    d = watch.MotionDetector((-0.1, -0.1, 0.6, 0.6), 0.004)
    assert score_after(d, with_blob(int(0.25 * W), int(0.25 * H))) > 0.004
    assert d.stay_mask((H, W))[0, 0] == 255 and d.stay_mask((H, W))[H - 1, W - 1] == 0
    whole = watch.MotionDetector((-1.0, -1.0, 0.5, 0.5), 0.004)  # a box with nothing in the frame: the whole frame
    assert score_after(whole, with_blob(int(0.75 * W), int(0.75 * H))) > 0


def test_log_rarely_logs_once_a_minute_per_key(monkeypatch, caplog):
    monkeypatch.setattr(watch, "_last_logged", {})
    clock = [1000.0]
    monkeypatch.setattr(watch.time, "monotonic", lambda: clock[0])
    with caplog.at_level("ERROR", logger="birdbrain.watch"):
        for _ in range(3):
            try:
                raise locked()
            except sqlite3.OperationalError:
                watch.log_rarely("frame", "couldn't process a frame")
        clock[0] += 61
        try:
            raise locked()
        except sqlite3.OperationalError:
            watch.log_rarely("frame", "couldn't process a frame")
    assert caplog.text.count("couldn't process a frame") == 2


class FakePush:
    """A push_public.sh process: running until told otherwise."""

    def __init__(self, running=True):
        self.running, self.signals = running, []

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.signals.append("term")
        self.running = False

    def kill(self):
        self.signals.append("kill")
        self.running = False

    def wait(self, timeout=None):
        return 0


def test_push_public_skips_a_running_push_but_stops_a_hung_one(monkeypatch):
    import subprocess, time

    monkeypatch.setenv("PUBLIC_DEST", "deploy@mirror:")
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kw: started.append(args) or FakePush())
    running = FakePush()
    monkeypatch.setattr(watch, "_push", running)
    monkeypatch.setattr(watch, "_push_started", time.monotonic() - 10)
    watch.push_public()
    assert started == [] and running.signals == []  # a recent push is left to finish
    monkeypatch.setattr(watch, "_push_started", time.monotonic() - watch.PUSH_TIMEOUT - 1)
    watch.push_public()
    assert running.signals == ["term"] and len(started) == 1 and started[0][0].endswith("scripts/push_public.sh")
    assert watch._push is not running and time.monotonic() - watch._push_started < 5
    watch._push.running = False  # done: the next visit starts a new one
    watch.push_public()
    assert len(started) == 2
    monkeypatch.delenv("PUBLIC_DEST")
    watch.push_public()
    assert len(started) == 2  # no mirror configured: nothing to do


CLEAR = {"finder": None, "motion": {"found": False, "why": "nothing", "source": "empty"},
         "tried": [{"finder": n, "found": False, "why": w} for n, w in
                   (("motion", "nothing"), ("blob", "no box"), ("yolox", "nothing"))]}
UNSURE = {"score": 0.05, "threshold": 0.02, "empty": False}  # past the Pi animal check, but only just


@pytest.mark.parametrize("trigger, night, check, finding, expected", [
    ("rim", False, UNSURE, CLEAR, True),               # the rain frames of 2026-11-10
    ("rim", False, {**UNSURE, "score": 0.9996}, CLEAR, False),  # visit 1104: a bird the check was sure of
    ("rim", False, {**UNSURE, "score": 0.3}, CLEAR, False),     # at the limit: Claude
    ("ground", False, UNSURE, CLEAR, False),           # the ground zone is Claude's (a squirrel in the shrub)
    ("water", False, UNSURE, CLEAR, False),
    ("rim", True, UNSURE, CLEAR, False),               # never at night
    ("rim", False, None, CLEAR, False),                # no verdict from the check
    ("rim", False, {"score": None, "empty": False}, CLEAR, False),
    ("rim", False, UNSURE, None, False),               # the finders never ran
    ("rim", False, UNSURE, {**CLEAR, "motion": {"found": False, "why": "nothing", "source": None}}, False),
])
def test_rain_on_water_needs_a_low_score_and_nothing_new_on_land(trigger, night, check, finding, expected):
    assert watch.rain_on_water(trigger, night, check, finding) is expected


def test_rain_on_water_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(watch.config, "RAIN_SCORE_MAX", 0.0)
    assert watch.rain_on_water("rim", False, UNSURE, CLEAR) is False


URL = "rtsp://camera.example:554/ch0"


def test_without_a_clip_the_burst_is_captured_as_it_always_was():
    assert watch.burst_command(URL, "/tmp/b") == [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", URL,
        "-vf", f"fps={config.BURST_FPS}", "-frames:v", str(config.BURST_FRAMES), "-q:v", "2", "/tmp/b/f%02d.jpg"]


def test_the_clip_rides_on_the_bursts_one_camera_connection():
    cmd = watch.burst_command(URL, "/tmp/b", 5)
    assert cmd.count("-i") == 1  # the camera serves only a few streams at once
    assert cmd[-1] == f"/tmp/b/{clips.CLIP_NAME}" and "/tmp/b/f%02d.jpg" in cmd
    clip = cmd[cmd.index("/tmp/b/f%02d.jpg") + 1:]
    assert clip[clip.index("-c:v") + 1] == "copy"  # as the camera sends it: nothing for the Pi to encode
    assert clip[clip.index("-t") + 1] == "5"
    # The stills are renamed into place when complete, so they can be read while the clip still records.
    assert cmd[cmd.index("-atomic_writing") + 1] == "1"


def test_the_stills_never_wait_for_the_clip_to_end():
    """With frame threads an older ffmpeg (5.1) keeps the last stills until it ends, 3.4 s later with a clip."""
    stills = (cmd := watch.burst_command(URL, "/tmp/b", 5))[:cmd.index("/tmp/b/f%02d.jpg")]
    assert stills[stills.index("-thread_type") + 1] == "slice"


def test_a_clip_never_has_sound():
    """The owner's rule: the microphone would pick up the neighbours."""
    cmd = watch.burst_command(URL, "/tmp/b", 5)
    assert "-an" in cmd[cmd.index("/tmp/b/f%02d.jpg") + 1:]
    assert not any(arg in ("-c:a", "-acodec") or arg.startswith("0:a") for arg in cmd)
