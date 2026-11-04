import sqlite3

import anthropic
import cv2
import httpx
import numpy as np
import pytest

from birdbrain import stay, watch

W, H = 640, 360
# A birdbath-like ring in the middle of the frame, with a rock spot inside the water.
RING = (0.5, 0.5, 0.2, 0.25, 0.15, 0.18)
SPOTS = ((0.42, 0.5, 0.03, 0.04),)


def frame(color=(90, 110, 80)) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = color
    return img


def jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def with_blob(x: int, y: int, r: int = 14) -> np.ndarray:
    img = frame()
    cv2.circle(img, (x, y), r, (30, 30, 30), -1)
    return img


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


WATER_SPOT = (int(0.56 * W), int(0.53 * H))  # inside the water, away from the rim and the rock spot


def water_detector():
    d = watch.MotionDetector(None, 0.02, RING, SPOTS)
    for _ in range(5):
        d.score(frame())
    return d


def run_frames(d, frames) -> list[bool]:
    hits = []
    for f in frames:
        d.score(f)
        hits.append(d.water_hit)
    return hits


def ripples(seed: int) -> np.ndarray:
    """Rain on the water: many small, faint-to-bright specks in new places every frame."""
    rng = np.random.default_rng(seed)
    img = frame()
    cx, cy = W // 2, H // 2
    for _ in range(60):
        x = int(cx + rng.uniform(-0.12, 0.12) * W)
        y = int(cy + rng.uniform(-0.15, 0.15) * H)
        cv2.circle(img, (x, y), int(rng.integers(1, 4)), (int(rng.integers(140, 230)),) * 3, -1)
    return img


def test_water_mask_is_the_inner_ellipse_minus_rocks():
    mask = watch.water_mask((H, W), RING, SPOTS)
    assert mask[WATER_SPOT[1], WATER_SPOT[0]] == 255
    assert mask[H // 2, int(0.42 * W)] == 0                 # the rock spot belongs to the rim zone
    assert mask[H // 2, int(0.5 * W + 0.175 * W)] == 0      # the rim itself


def test_ripples_and_raindrops_never_trigger_the_water_zone():
    assert not any(run_frames(water_detector(), [ripples(i) for i in range(60)]))


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


@pytest.fixture
def pipeline(monkeypatch, conn):
    """handle_event with the camera, the local animal check and Claude faked."""
    burst = [cv2.imencode(".jpg", frame())[1].tobytes()] * 3
    monkeypatch.setattr(watch, "capture_burst", lambda: burst)
    asked = []

    def fake_identify(*args, **kwargs):
        from birdbrain.identify import Result, Sighting
        asked.append(1)
        return Result(Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes="", circle=None, also=[]),
                      "claude-haiku-5-5", False, False, {})

    monkeypatch.setattr(watch.identify, "identify", fake_identify)

    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    monkeypatch.setattr(watch.classifier, "box_embedding", lambda jpeg, zones, trigger: None)

    def set_detector(animal):
        """animal=True: the animal check sees something; False: it calls the frame empty."""
        monkeypatch.setattr(watch.presence, "check", lambda conn, emb, view=None: {
            "model": "bioclip-presence", "score": 0.9 if animal else 0.1, "threshold": 0.4, "empty": not animal})

    return conn, asked, set_detector


def test_probe_that_sees_nothing_is_discarded_without_calling_claude(pipeline):
    conn, asked, set_detector = pipeline
    set_detector(False)
    assert watch.handle_event(probe=True, trigger="water") == watch.PROBE_EMPTY
    assert asked == [] and watch.db.count_visits(conn) == 0 and watch.db.count_visits(conn, false_triggers=True) == 0


def test_probe_that_sees_an_animal_goes_to_claude_and_is_stored(pipeline):
    conn, asked, set_detector = pipeline
    set_detector(True)
    assert watch.handle_event(probe=True, trigger="water") == "bird"
    assert asked == [1]
    raw = watch.json.loads(watch.db.latest_visit(conn)["raw_json"])
    assert (raw["trigger"], raw["probe"]) == ("water", True)


def test_false_trigger_keeps_its_whole_burst(pipeline, monkeypatch):
    conn, asked, set_detector = pipeline
    set_detector(True)
    from birdbrain.identify import Result, Sighting
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: Result(
        Sighting(visitor="none", species="", count=0, confidence="high", notes="", circle=None, also=[]), "claude-haiku-5-5", False, False, {}))
    assert watch.handle_event() == "none"
    visit = watch.db.latest_visit(conn, real_only=False)
    raw = watch.json.loads(visit["raw_json"])
    assert raw["burst"]["count"] == 3 and raw["burst"]["best"] == 0 and len(raw["burst"]["sharpness"]) == 3
    assert len(watch.images.burst_paths(visit["image_path"])) == 3


def test_claude_is_told_whether_the_frame_is_infrared(pipeline, monkeypatch):
    conn, asked, set_detector = pipeline
    set_detector(True)
    from birdbrain.identify import Result, Sighting
    seen = []

    def identify(jpeg, *a, **k):
        seen.append(k.get("night"))
        return Result(Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes="", circle=None,
                               also=[]), "claude-haiku-5-5", False, False, {})

    monkeypatch.setattr(watch.identify, "identify", identify)
    for night in (True, False):
        monkeypatch.setattr(watch, "is_night", lambda jpeg, night=night: night)
        watch.handle_event()
    assert seen == [True, False]


def test_alerts_get_claudes_confidence_and_the_review_flag(pipeline, monkeypatch):
    conn, asked, set_detector = pipeline
    set_detector(True)
    from birdbrain.identify import Result, Sighting
    also = [{"visitor": "bird", "species": "Bushtit", "count": 2, "confidence": "medium", "circle": None}]
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: Result(
        Sighting(visitor="bird", species="Varied Thrush", count=1, confidence="low", notes="", circle=None, also=[]),
        "claude-sonnet-5-5", True, True, {"also": also}))
    watch.db.set_value(conn, "ntfy_topic", "t")
    sent, reasons = [], []
    monkeypatch.setattr(watch.alert, "send", lambda topic, title, *a: sent.append(title))
    real_reason = watch.alert.reason
    monkeypatch.setattr(watch.alert, "reason", lambda *a: reasons.append(a[1:]) or real_reason(*a))
    assert watch.handle_event() == "bird"
    assert reasons[0][:2] == ("bird", "Varied Thrush") and reasons[0][3:] == ("low", True)
    assert sent == ["Possible new species: Varied Thrush (needs your check)"]


def test_real_visit_keeps_only_the_best_frame(pipeline):
    conn, asked, set_detector = pipeline
    set_detector(True)
    assert watch.handle_event() == "bird"
    visit = watch.db.latest_visit(conn)
    assert "burst" not in watch.json.loads(visit["raw_json"])
    assert watch.images.burst_paths(visit["image_path"]) == []


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


def test_false_triggers_are_saved_at_the_photo_size_from_settings(pipeline, monkeypatch):
    conn, asked, set_detector = pipeline
    set_detector(False)
    from birdbrain.identify import Result, Sighting
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: Result(
        Sighting(visitor="none", species=None, count=0, confidence="high", notes="", circle=None, also=[]), "claude-haiku-5-5", False, False, {}))
    saved = []
    monkeypatch.setattr(watch.images, "store_frame", lambda jpeg, when, **kw: saved.append(kw) or ("a.jpg", "a_t.jpg"))
    watch.retention.save_options(conn, limit=0, photo="small")
    watch.handle_event()
    assert saved[-1] == {"max_edge": 640, "quality": 70}


# --- Ground zone: critters passing by under and around the bath ---

GROUND = ((0.0, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0))  # the bottom of the frame; the bath's lower edge dips in
GROUND_Y = int(0.88 * H)


def night_frame() -> np.ndarray:
    return frame((100, 100, 100))  # IR: no color


def walker(base, x: int, r: int, y: int = GROUND_Y) -> np.ndarray:
    img = base.copy()
    cv2.circle(img, (x, y), r, (20, 20, 20), -1)
    return img


def ground_detector(base) -> watch.MotionDetector:
    d = watch.MotionDetector(None, 0.02, RING, SPOTS, GROUND)
    for _ in range(5):
        d.score(base)
    return d


def ground_hits(d, frames) -> list[bool]:
    hits = []
    for f in frames:
        d.score(f)
        hits.append(d.ground_hit)
    return hits


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


def test_small_things_count_on_the_ground_by_day_but_not_at_night():
    day, night = frame(), night_frame()
    # A junco-sized blob (~0.004 of the frame): above the day minimum, under the night one.
    assert any(ground_hits(ground_detector(day), [walker(day, 100 + 3 * i, r=17) for i in range(12)]))
    assert not any(ground_hits(ground_detector(night), [walker(night, 100 + 3 * i, r=17) for i in range(12)]))


def test_flicker_and_teleporting_blobs_are_not_animals():
    base = night_frame()
    flicker = ([walker(base, 200, r=28)] + [base] * 2) * 5        # a moth passing the IR light, or a leaf
    assert not any(ground_hits(ground_detector(base), flicker))
    jumps = [walker(base, 60 if i % 2 else 560, r=28) for i in range(12)]  # far apart frame to frame
    assert not any(ground_hits(ground_detector(base), jumps))


def test_the_whole_ground_changing_at_once_is_light_not_an_animal():
    base = night_frame()
    lit = base.copy()
    lit[int(0.6 * H):] = (200, 200, 200)                          # headlights, or the dusk IR switch
    assert not any(ground_hits(ground_detector(base), [lit] * 12))


def test_the_water_zone_sleeps_at_night():
    base = night_frame()
    d = watch.MotionDetector(None, 0.02, RING, SPOTS)
    for _ in range(5):
        d.score(base)
    assert not any(run_frames(d, [walker(base, WATER_SPOT[0], 12, WATER_SPOT[1])] * 12))
    assert any(run_frames(water_detector(), [with_blob(*WATER_SPOT, r=12)] * 12))  # same bird by day


def test_ground_calls_have_their_own_daily_count(conn, add_visit):
    add_visit(raw_json=watch.json.dumps({"trigger": "rim"}))
    add_visit(raw_json=watch.json.dumps({"trigger": "ground"}))
    add_visit(raw_json=watch.json.dumps({"trigger": "ground"}))
    add_visit(raw_json=None)                                                         # --once
    add_visit(model="yolox-tiny", raw_json=watch.json.dumps({"trigger": "ground"}))  # settled locally: free
    assert watch.calls_today(ground=False) == 2
    assert watch.calls_today(ground=True) == 2


def test_ground_events_use_the_ground_crop(pipeline, monkeypatch):
    conn, asked, set_detector = pipeline
    views = []
    set_detector(True)
    monkeypatch.setattr(watch.classifier, "box_embedding", lambda jpeg, z, trigger: views.append((trigger, z)))
    monkeypatch.setattr(watch.config, "GROUND_CROP", (0.0, 0.5, 1.0, 0.5))
    watch.handle_event(trigger="ground")
    raw = watch.json.loads(watch.db.latest_visit(conn)["raw_json"])
    assert raw["trigger"] == "ground"
    # The zones in use are stored with the visit, for the visit page's drawing.
    assert watch.zones.Zones.from_dict(raw["zones"]).ground_crop == (0.0, 0.5, 1.0, 0.5)
    watch.handle_event(trigger="rim")
    assert [t for t, _ in views] == ["ground", "rim"]
    assert views[0][1].ground_crop == (0.0, 0.5, 1.0, 0.5) and views[1][1].crop == watch.config.CROP_ROI


def test_ground_results_never_put_the_birdbath_on_cooldown(monkeypatch, conn):
    w = watch.Watcher()
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: "none")
    w._work(0.0, probe=False, trigger="ground")
    w._work(0.0, probe=False, trigger="ground")
    assert w.cooldown_until == 0.0 and w.backoff_until == 0.0 and w.false_streak == 0
    assert w.ground_streak == 2 and w.ground_backoff_until > watch.time.monotonic() + watch.config.COOLDOWN_NONE
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: watch.PROBE_EMPTY)
    w._work(0.0, probe=True, trigger="ground")  # an empty probe keeps the backoff going
    assert w.ground_streak == 2 and w.ground_backoff_until > watch.time.monotonic()
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: "raccoon")
    w._work(0.0, probe=False, trigger="ground")
    assert w.ground_streak == 0 and w.cooldown_until == 0.0


def test_ground_limit_pauses_only_the_ground(monkeypatch, conn):
    w = watch.Watcher()
    monkeypatch.setattr(watch.config, "GROUND_MAX_CALLS_PER_DAY", 0)
    started = []

    class FakeThread:
        def __init__(self, **kwargs):
            started.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(watch.threading, "Thread", FakeThread)
    w._maybe_trigger(0.0, trigger="ground")
    assert started == [] and w.ground_until > watch.time.monotonic() and w.cooldown_until == 0.0
    w._maybe_trigger(0.0, trigger="rim")
    assert len(started) == 1


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


def test_a_probe_without_a_verdict_is_dropped_not_sent_to_claude(pipeline, monkeypatch):
    conn, asked, set_detector = pipeline
    monkeypatch.setattr(watch.presence, "check", lambda conn, emb, view=None: {"model": "bioclip-presence", "score": None, "empty": False})
    assert watch.handle_event(probe=True, trigger="ground") == watch.PROBE_UNCHECKED
    monkeypatch.setattr(watch.classifier, "box_embedding", raise_locked)  # the embedding failed
    assert watch.handle_event(probe=True, trigger="rim") == watch.PROBE_UNCHECKED
    assert asked == [] and watch.db.count_visits(conn, false_triggers=True) == 0
    assert watch.handle_event(probe=False, trigger="rim") == "bird"  # a full check still goes to Claude


@pytest.fixture
def probes_on(monkeypatch):
    """A trained animal check with its embedding model loaded, as a rainy day with a well-taught Pi has."""
    monkeypatch.setattr(watch.presence, "trained", lambda: True)
    monkeypatch.setattr(watch.classifier, "usable", lambda: True)
    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")


def test_probe_ready_needs_the_classifier_on_and_a_trained_check(monkeypatch, conn, probes_on):
    assert watch.probe_ready() is True
    monkeypatch.setattr(watch.presence, "trained", lambda: False)  # fewer than 10 visitors / 30 empty frames
    assert watch.probe_ready() is False
    monkeypatch.setattr(watch.presence, "trained", lambda: True)
    monkeypatch.setattr(watch.classifier, "usable", lambda: False)  # CLASSIFIER=off, or its download failed
    assert watch.probe_ready() is False
    monkeypatch.setattr(watch.classifier, "usable", lambda: True)
    monkeypatch.setattr(watch, "presence_mode", lambda: "off")
    assert watch.probe_ready() is False


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


def doubling_calls(seconds: int) -> int:
    """How many Claude calls the plain doubling backoff allows in `seconds` of steady false triggers."""
    t = n = streak = 0
    while t < seconds:
        n, streak = n + 1, streak + 1
        t += watch.config.false_trigger_wait(streak)
    return n


@pytest.mark.parametrize("why", ["classifier off", "check untrained"])
def test_rain_without_a_verdict_costs_no_more_than_the_doubling_backoff(pipeline, monkeypatch, why):
    """An hour of rain with motion on every frame. Before, probes with nothing filtering them went to Claude
    every COOLDOWN_NONE: 180 calls an hour, the day's MAX_CALLS_PER_DAY gone by lunch."""
    conn, asked, set_detector = pipeline
    if why == "classifier off":
        monkeypatch.setattr(watch.config, "CLASSIFIER", "off")  # presence keeps its default "shadow"
    else:
        monkeypatch.setattr(watch.presence, "check", lambda conn, emb, view=None: {"model": "bioclip-presence", "score": None,
                                                                         "empty": False})
    from birdbrain.identify import Result, Sighting
    monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: asked.append(1) or Result(
        Sighting(visitor="none", species=None, count=0, confidence="high", notes="", circle=None, also=[]), "claude-haiku-5-5", False, False, {}))
    clock = [1000.0]
    monkeypatch.setattr(watch.time, "monotonic", lambda: clock[0])
    w = watch.Watcher()
    for _ in range(3600):
        action = watch.motion_action(clock[0], w.cooldown_until, w.backoff_until, watch.probe_ready())
        if action != "wait":
            w._work(0.05, probe=action == "probe", trigger="rim")
        clock[0] += 1
    assert len(asked) == doubling_calls(3600) <= 16


def test_a_probe_with_no_verdict_falls_back_to_the_doubling_backoff(monkeypatch, conn, probes_on):
    w = watch.Watcher()
    w.false_streak, w.ground_streak = 3, 3
    w.backoff_until = w.ground_backoff_until = watch.time.monotonic() + watch.config.FULL_CHECK_EVERY
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: watch.PROBE_UNCHECKED)
    for trigger, until, backoff, streak in (("rim", "cooldown_until", "backoff_until", "false_streak"),
                                            ("ground", "ground_until", "ground_backoff_until", "ground_streak")):
        w._work(0.0, probe=True, trigger=trigger)
        now = watch.time.monotonic()
        assert getattr(w, streak) == 4 and w.stay_pending is None
        # No probes every COOLDOWN_NONE: the next trigger is a full one after the doubling wait.
        assert getattr(w, until) == getattr(w, backoff) == pytest.approx(now + watch.config.false_trigger_wait(4), abs=1)
        assert watch.motion_action(getattr(w, until), getattr(w, until), getattr(w, backoff), True) == "full"
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: "none")
    w._work(0.0, probe=False, trigger="rim")  # a verdict-less run over, the trained check probes again
    assert w.false_streak == 5 and w.backoff_until == pytest.approx(w.cooldown_until - watch.config.COOLDOWN_NONE
                                                                    + watch.config.FULL_CHECK_EVERY, abs=1)


# --- Zones and boxes stay inside the frame ---

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


# --- The ground zone answers only to its own cooldown; timers run on the monotonic clock ---

@pytest.fixture
def fake_threads(monkeypatch):
    """Capture threads never start; the triggers they would have run are listed as (trigger, probe)."""
    started = []

    class FakeThread:
        def __init__(self, target=None, args=(), **kwargs):
            started.append((args[2], args[1]))

        def start(self):
            pass

    monkeypatch.setattr(watch.threading, "Thread", FakeThread)
    return started


def ground_motion(d: watch.MotionDetector):
    """A detector whose next frames show an animal on the ground and nothing at the bath."""
    def score(frame):
        d.ground_hit, d.water_hit = True, False
        return 0.0
    return score


def test_a_ground_trigger_fires_while_the_bath_is_cooling_down_or_over_its_limit(monkeypatch, conn, fake_threads):
    w = watch.Watcher()
    monkeypatch.setattr(w.detector, "score", ground_motion(w.detector))
    w.cooldown_until = watch.time.monotonic() + 300  # the bath backing off after rainy false alarms
    w.step(frame(), WARMUP_FRAMES_DONE)
    assert fake_threads == [("ground", False)]
    w.busy.clear()
    monkeypatch.setattr(watch.config, "MAX_CALLS_PER_DAY", 0)  # the bath's daily limit reached...
    w._maybe_trigger(0.5, trigger="rim")
    assert len(fake_threads) == 1 and w.cooldown_until > watch.time.monotonic()
    w.step(frame(), WARMUP_FRAMES_DONE)  # ...and the ground still fires, on its own budget
    assert fake_threads == [("ground", False)] * 2
    w.busy.clear()
    w.ground_until = watch.time.monotonic() + 300  # its own cooldown is what holds it
    w.step(frame(), WARMUP_FRAMES_DONE)
    assert len(fake_threads) == 2


WARMUP_FRAMES_DONE = watch.WARMUP_FRAMES + 1


def test_cooldowns_run_on_the_monotonic_clock_not_the_wall_clock(monkeypatch, conn, fake_threads):
    mono, wall = [5000.0], [1_700_000_000.0]
    monkeypatch.setattr(watch.time, "monotonic", lambda: mono[0])
    monkeypatch.setattr(watch.time, "time", lambda: wall[0])
    w = watch.Watcher()
    monkeypatch.setattr(w.detector, "score", lambda frame: 0.5)  # rim motion on every frame
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: "none")
    w._work(0.0, probe=False, trigger="rim")
    assert w.cooldown_until == 5000.0 + watch.config.COOLDOWN_NONE

    def rim_motion() -> int:
        before = len(fake_threads)
        for _ in range(watch.config.TRIGGER_FRAMES):
            w.step(frame(), WARMUP_FRAMES_DONE)
        return len(fake_threads) - before

    wall[0] += 3600  # the wall clock leaps an hour ahead (NTP after boot): the cooldown must not end early
    mono[0] += 1
    assert rim_motion() == 0
    wall[0] -= 7200  # ...nor must a step back stretch it
    mono[0] += watch.config.COOLDOWN_NONE
    assert rim_motion() == 1


# --- Never lose a visit: the watcher must survive Claude and database errors with the visit stored ---

def locked():
    return sqlite3.OperationalError("database is locked")


def raise_locked(*args, **kwargs):
    raise locked()


def test_work_clears_busy_even_when_the_cooldown_step_raises(monkeypatch, conn):
    w = watch.Watcher()
    w.busy.set()
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: "none")
    monkeypatch.setattr(watch, "probe_ready", raise_locked)
    w._work(0.0, probe=False, trigger="rim")
    assert not w.busy.is_set()


def test_probe_ready_falls_back_to_no_probes_on_a_locked_database(monkeypatch, conn, probes_on):
    w = watch.Watcher()
    w.busy.set()
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: "none")
    monkeypatch.setattr(watch, "presence_mode", raise_locked)
    assert watch.probe_ready() is False
    w._work(0.0, probe=False, trigger="rim")
    assert not w.busy.is_set() and w.false_streak == 1 and w.cooldown_until > watch.time.monotonic()
    assert w.backoff_until == w.cooldown_until  # no probes without the animal check: just wait


def api_errors():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return [anthropic.APIConnectionError(request=req),
            anthropic.RateLimitError("rate limited", response=httpx.Response(429, request=req), body=None)]


@pytest.mark.parametrize("error", api_errors(), ids=["connection", "rate limit"])
def test_claude_error_keeps_the_visit_for_review_without_touching_the_streak(pipeline, monkeypatch, error):
    conn, asked, set_detector = pipeline
    set_detector(True)

    def failing_identify(*args, **kwargs):
        raise error

    monkeypatch.setattr(watch.identify, "identify", failing_identify)
    alerts = []
    monkeypatch.setattr(watch.alert, "reason", lambda *a: alerts.append(a) or "New species")
    w = watch.Watcher()
    w.false_streak = 2
    w.empty = frame()[:, :, 0]
    w.busy.set()
    w._work(0.01, probe=False, trigger="rim")
    assert not w.busy.is_set()
    visit = watch.db.latest_visit(conn)
    assert (visit["visitor"], visit["species"], visit["needs_review"], visit["model"]) == ("other", None, 1, "claude-haiku-5-5")
    raw = watch.json.loads(visit["raw_json"])
    assert raw["calls"]["error"].startswith(type(error).__name__)
    assert w.false_streak == 2 and w.backoff_until == 0.0          # not a false trigger...
    assert w.stay_pending is None and alerts == []                 # ...and not a visit to alert about or follow
    assert w.cooldown_until > watch.time.monotonic()
    assert watch.calls_today() == 1                                 # the failed call still counts toward the budget


def test_unanswered_ground_trigger_waits_without_a_ground_streak(monkeypatch, conn):
    w = watch.Watcher()
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: watch.UNANSWERED)
    w._work(0.0, probe=False, trigger="ground")
    assert w.ground_streak == 0 and w.ground_until > watch.time.monotonic() and w.cooldown_until == 0.0


@pytest.mark.parametrize("effect", ["note_stored", "push_public", "prune"])
def test_an_error_after_the_save_still_returns_the_stored_visitor(pipeline, monkeypatch, effect):
    conn, asked, set_detector = pipeline
    set_detector(True)
    if effect == "prune":
        from birdbrain.identify import Result, Sighting
        monkeypatch.setattr(watch.identify, "identify", lambda *a, **k: Result(
            Sighting(visitor="none", species=None, count=0, confidence="high", notes="", circle=None, also=[]), "claude-haiku-5-5", False, False, {}))
        monkeypatch.setattr(watch.retention, "prune", raise_locked)
        expected = "none"
    elif effect == "note_stored":
        monkeypatch.setattr(watch.retention, "note_stored", raise_locked)
        expected = "bird"
    else:
        monkeypatch.setenv("PUBLIC_DEST", "pi@mirror:/srv")
        monkeypatch.setattr(watch.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("not executable")))
        expected = "bird"
    assert watch.handle_event() == expected
    assert watch.db.latest_visit(conn, real_only=False)["visitor"] == expected
    assert watch.last_visit_id is not None


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


class Stop(BaseException):
    """Breaks out of Watcher.run's forever loop when the fake camera is opened a second time."""


@pytest.fixture
def stalled_camera(monkeypatch):
    """A substream that delivers 5 frames, then nothing: every read takes a second and fails. Opening it a
    second time (the reconnect) raises Stop. Returns the list of captures opened."""
    opened = []
    clock = [100.0]

    class FakeCap:
        def __init__(self, *args):
            if opened:
                raise Stop
            opened.append(self)
            self.args = args
            self.reads = 0

        def isOpened(self):
            return True

        def read(self):
            self.reads += 1
            clock[0] += 1.0
            return (True, frame()) if self.reads <= 5 else (False, None)

        def release(self):
            pass

    monkeypatch.setattr(watch.cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(watch.time, "sleep", lambda s: None)
    monkeypatch.setattr(watch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(watch, "_last_logged", {})
    return opened


def test_frame_loop_survives_a_locked_database(monkeypatch, conn, stalled_camera):
    w = watch.Watcher()
    asked = []
    monkeypatch.setattr(watch.zones, "current", lambda: asked.append(1) or raise_locked())
    with pytest.raises(Stop):
        w.run()
    assert len(asked) == 5  # every frame was tried; none of the errors took the loop down


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


def test_reconnects_after_30_s_without_a_frame(conn, stalled_camera):
    """Counting 50 failed reads was 25 minutes blind: FFmpeg blocks each read of a stalled stream for ~30 s."""
    with pytest.raises(Stop):
        watch.Watcher().run()
    cap = stalled_camera[0]
    assert abs(cap.reads - (5 + watch.STALL_SECONDS)) <= 1
    url, backend, *params = cap.args
    assert backend == cv2.CAP_FFMPEG
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):  # the timeouts are set at open time, where the build has them
        assert params == [[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, watch.STREAM_TIMEOUT_MS,
                           cv2.CAP_PROP_READ_TIMEOUT_MSEC, watch.STREAM_TIMEOUT_MS]]
