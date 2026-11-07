"""Unit tests of birdbrain/config: pure functions, no files, database or network."""

import pytest

from birdbrain import config


def test_parse_box():
    assert config.parse_box("0.1, 0.2, 0.3, 0.4") == (0.1, 0.2, 0.3, 0.4)
    assert config.parse_box("") is None
    assert config.parse_box("   ") is None


@pytest.mark.parametrize("raw", ["0.1,0.2,0.3", "a,b,c,d", "0.1,0.2,0.3,0.4,0.5"])
def test_parse_box_rejects_bad_values_with_the_setting_name(raw):
    with pytest.raises(ValueError, match="CROP_ROI"):
        config.parse_box(raw, "CROP_ROI")


def test_parse_ring_accepts_6_or_7_numbers():
    assert config.parse_ring("1,2,3,4,5,6") == (1, 2, 3, 4, 5, 6)
    assert config.parse_ring("1,2,3,4,5,6,7") == (1, 2, 3, 4, 5, 6, 7)
    assert config.parse_ring("") is None
    with pytest.raises(ValueError):
        config.parse_ring("1,2,3,4,5")


def test_parse_spots():
    assert config.parse_spots("") == ()
    assert config.parse_spots("0.1,0.2,0.3,0.4; 0.5,0.6,0.7,0.8;") == ((0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.7, 0.8))
    with pytest.raises(ValueError, match="MOTION_SPOTS"):
        config.parse_spots("0.1,0.2,0.3", "MOTION_SPOTS")


def test_rtsp_url_encodes_special_characters_in_the_password():
    url = config.rtsp_url("ch1")
    assert url.endswith("@birdbrain-cam.home:554/ch1")
    assert "p%40ss%23word%21" in url
    assert "p@ss#" not in url


def test_rtsp_url_defaults_to_the_configured_main_stream(monkeypatch):
    monkeypatch.setattr(config, "RTSP_MAIN", "stream1")
    monkeypatch.setattr(config, "RTSP_PORT", 8554)
    assert config.rtsp_url().endswith(":8554/stream1")
    assert config.rtsp_url("/live/sub").endswith(":8554/live/sub")


def test_false_trigger_wait_doubles_up_to_the_cap(monkeypatch):
    monkeypatch.setattr(config, "COOLDOWN_NONE", 20)
    monkeypatch.setattr(config, "COOLDOWN_NONE_MAX", 300)
    assert [config.false_trigger_wait(n) for n in range(1, 7)] == [20, 40, 80, 160, 300, 300]
    assert config.false_trigger_wait(0) == 20


def test_parse_polygon_and_its_box():
    points = config.parse_polygon("0.1,0.5; 0.9,0.5; 0.9,1; 0.1,1;")
    assert points == ((0.1, 0.5), (0.9, 0.5), (0.9, 1), (0.1, 1))
    assert config.polygon_box(points) == pytest.approx((0.1, 0.5, 0.8, 0.5))
    assert config.parse_polygon("") == () and config.polygon_box(()) is None
    with pytest.raises(ValueError, match="GROUND_ZONE"):
        config.parse_polygon("0.1,0.5;0.9,0.5", "GROUND_ZONE")   # two points is a line
    with pytest.raises(ValueError, match="GROUND_ZONE"):
        config.parse_polygon("0.1,0.5,0.2;0.9,0.5;0.9,1", "GROUND_ZONE")


# --- the finder chain's knobs (FINDERS, FINDER_YOLOX_MIN_SCORE) -------------------------------------------------

def _fresh_config(monkeypatch, **env) -> dict:
    """config.py run again in a namespace of its own with `env` set (the imported module is left alone), so what
    it does at import can be checked: the parsing and the checks that raise."""
    import runpy
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return runpy.run_path(config.__file__, run_name="birdbrain.config")


def test_finders_default_to_motion_blob_yolox_in_that_order():
    assert tuple(config.FINDERS) == ("motion", "blob", "yolox")
    assert config.FINDER_YOLOX_MIN_SCORE == 0.15


def test_finders_come_from_the_env_as_a_comma_list(monkeypatch):
    ns = _fresh_config(monkeypatch, FINDERS="yolox, motion", FINDER_YOLOX_MIN_SCORE="0.3")
    assert tuple(ns["FINDERS"]) == ("yolox", "motion") and ns["FINDER_YOLOX_MIN_SCORE"] == 0.3
    assert tuple(_fresh_config(monkeypatch, FINDERS="motion")["FINDERS"]) == ("motion",)
    monkeypatch.delenv("FINDERS")
    monkeypatch.delenv("FINDER_YOLOX_MIN_SCORE")
    ns = _fresh_config(monkeypatch)
    assert tuple(ns["FINDERS"]) == ("motion", "blob", "yolox") and ns["FINDER_YOLOX_MIN_SCORE"] == 0.15


def test_an_unknown_finder_name_is_refused_at_import(monkeypatch):
    with pytest.raises(ValueError, match="FINDERS"):
        _fresh_config(monkeypatch, FINDERS="motion,sonar")


def test_second_frame_min_defaults_to_0_8_and_off_or_0_disables_it(monkeypatch):
    assert config.SECOND_FRAME_MIN == 0.8
    assert _fresh_config(monkeypatch, SECOND_FRAME_MIN="0.85")["SECOND_FRAME_MIN"] == 0.85
    assert _fresh_config(monkeypatch, SECOND_FRAME_MIN="")["SECOND_FRAME_MIN"] == 0.8  # a blank .env line keeps the default
    assert _fresh_config(monkeypatch, SECOND_FRAME_MIN=" ")["SECOND_FRAME_MIN"] == 0.8
    assert _fresh_config(monkeypatch, SECOND_FRAME_MIN="off")["SECOND_FRAME_MIN"] == 0.0
    assert _fresh_config(monkeypatch, SECOND_FRAME_MIN="0")["SECOND_FRAME_MIN"] == 0.0
    with pytest.raises(ValueError, match="SECOND_FRAME_MIN"):
        _fresh_config(monkeypatch, SECOND_FRAME_MIN="1.5")


def test_second_frame_min_defaults_to_the_bowl_bar_and_is_a_fraction():
    # The shadow second-frame look runs when the tight look's top score is at least this (and under its own bar).
    assert config.SECOND_FRAME_MIN == 0.8
    assert 0 <= config.SECOND_FRAME_MIN < config.GATE_MIN_SCORE_TIGHT
