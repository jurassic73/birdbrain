import cv2
import numpy as np
import pytest

from birdbrain import watch

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
        return Result(Sighting(visitor="bird", species="Bushtit", count=1, confidence="high", notes=""),
                      "claude-haiku-5-5", False, False, {})

    monkeypatch.setattr(watch.identify, "identify", fake_identify)

    monkeypatch.setattr(watch, "presence_mode", lambda: "shadow")
    monkeypatch.setattr(watch.classifier, "box_embedding", lambda jpeg, box: None)

    def set_detector(animal):
        """animal=True: the animal check sees something; False: it calls the frame empty."""
        monkeypatch.setattr(watch.presence, "check", lambda conn, emb: {
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
        Sighting(visitor="none", species="", count=0, confidence="high", notes=""), "claude-haiku-5-5", False, False, {}))
    assert watch.handle_event() == "none"
    visit = watch.db.latest_visit(conn, real_only=False)
    raw = watch.json.loads(visit["raw_json"])
    assert raw["burst"]["count"] == 3 and raw["burst"]["best"] == 0 and len(raw["burst"]["sharpness"]) == 3
    assert len(watch.images.burst_paths(visit["image_path"])) == 3


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
        Sighting(visitor="none", species=None, count=0, confidence="high", notes=""), "claude-haiku-5-5", False, False, {}))
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
    crops = []
    set_detector(True)
    monkeypatch.setattr(watch.classifier, "box_embedding", lambda jpeg, crop: crops.append(crop))
    monkeypatch.setattr(watch.config, "GROUND_CROP", (0.0, 0.5, 1.0, 0.5))
    watch.handle_event(trigger="ground")
    assert watch.json.loads(watch.db.latest_visit(conn)["raw_json"])["trigger"] == "ground"
    watch.handle_event(trigger="rim")
    assert crops == [(0.0, 0.5, 1.0, 0.5), watch.config.CROP_ROI]


def test_ground_results_never_put_the_birdbath_on_cooldown(monkeypatch, conn):
    w = watch.Watcher()
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: "none")
    w._work(0.0, probe=False, trigger="ground")
    w._work(0.0, probe=False, trigger="ground")
    assert w.cooldown_until == 0.0 and w.backoff_until == 0.0 and w.false_streak == 0
    assert w.ground_streak == 2 and w.ground_backoff_until > watch.time.time() + watch.config.COOLDOWN_NONE
    monkeypatch.setattr(watch, "handle_event", lambda *a, **k: watch.PROBE_EMPTY)
    w._work(0.0, probe=True, trigger="ground")  # an empty probe keeps the backoff going
    assert w.ground_streak == 2 and w.ground_backoff_until > watch.time.time()
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
    assert started == [] and w.ground_until > watch.time.time() and w.cooldown_until == 0.0
    w._maybe_trigger(0.0, trigger="rim")
    assert len(started) == 1


def test_long_runs_of_false_triggers_probe_for_half_an_hour():
    short = watch.after_false(1000.0, 1, 20, can_probe=True)
    long = watch.after_false(1000.0, watch.config.PROBE_AFTER, 300, can_probe=True)
    assert short == (1000.0 + watch.config.COOLDOWN_NONE, 1020.0)
    assert long == (1000.0 + watch.config.COOLDOWN_NONE, 1000.0 + watch.config.FULL_CHECK_EVERY)
    assert watch.after_false(1000.0, 9, 300, can_probe=False) == (1300.0, 1300.0)  # no animal check: just wait


def test_probe_passes_unless_the_animal_check_calls_it_empty():
    assert watch.probe_passes({"empty": False, "score": 0.9})
    assert not watch.probe_passes({"empty": True, "score": 0.1})
    assert watch.probe_passes(None)  # check off or failed: Claude decides
    assert watch.probe_passes({"empty": False, "score": None})  # not trained yet


def test_probe_with_an_untrained_check_goes_to_claude(pipeline, monkeypatch):
    conn, asked, set_detector = pipeline
    monkeypatch.setattr(watch.presence, "check", lambda conn, emb: {"model": "bioclip-presence", "score": None, "empty": False})
    assert watch.handle_event(probe=True, trigger="ground") == "bird"
    assert asked == [1]
