"""Pure pieces of the Hailo shadow detector: letterbox mapping, NMS parsing, zone tagging and the tracker."""
import numpy as np
import pytest

from birdbrain import zones
from birdbrain.hailo import geometry as g

SUB = (360, 640, 3)  # the substream frame


def test_letterbox_fits_the_substream_frame_at_the_top_left():
    frame = np.zeros(SUB, np.uint8)
    frame[:] = (255, 0, 0)  # BGR blue
    batch, ratio = g.letterbox(frame)
    assert batch.shape == (1, 640, 640, 3) and batch.dtype == np.uint8
    assert ratio == 1.0
    assert tuple(batch[0, 0, 0]) == (0, 0, 255)      # RGB now
    assert tuple(batch[0, 500, 0]) == (114, 114, 114)  # grey padding below the picture


def test_letterbox_scales_a_big_frame_down():
    batch, ratio = g.letterbox(np.zeros((1080, 1920, 3), np.uint8))
    assert ratio == pytest.approx(640 / 1920)
    assert batch.shape == (1, 640, 640, 3)


def _nms(**per_class):
    """A HailoRT NMS-by-class output: 80 arrays, each (n, 5) rows y0, x0, y1, x1, score on the letterbox."""
    out = [np.zeros((0, 5), np.float32) for _ in range(80)]
    for label, rows in per_class.items():
        out[g.COCO.index(label)] = np.array(rows, np.float32)
    return out


def test_parse_nms_maps_letterbox_fractions_back_to_the_frame():
    # A bird covering the letterbox's top-left quarter: on a 640x360 frame (ratio 1) that is the left half, and
    # the top 160/360 of the height.
    out = _nms(bird=[[0.0, 0.0, 0.25, 0.5, 0.9]])
    found = g.parse_nms(out, 1.0, SUB, 0.3)
    assert len(found) == 1
    d = found[0]
    assert d.label == "bird" and d.score == 0.9 and d.animal
    assert d.box == (0.0, 0.0, 0.5, pytest.approx(160 / 360, abs=1e-3))


def test_parse_nms_drops_low_scores_and_sorts_best_first():
    out = _nms(bird=[[0.1, 0.1, 0.2, 0.2, 0.35]], cat=[[0.3, 0.3, 0.4, 0.4, 0.8]], dog=[[0.5, 0.5, 0.6, 0.6, 0.2]])
    found = g.parse_nms(out, 1.0, SUB, 0.3)
    assert [d.label for d in found] == ["cat", "bird"]


def test_parse_nms_clips_boxes_that_reach_into_the_padding():
    out = _nms(bird=[[0.4, 0.4, 0.9, 0.9, 0.5]])  # y1 = 0.9 * 640 = 576 px, below the 360 px picture
    (d,) = g.parse_nms(out, 1.0, SUB, 0.3)
    assert d.box[3] == 1.0 and d.box[2] == pytest.approx(0.9)


def test_parse_nms_handles_empty_output():
    assert g.parse_nms([np.zeros((0, 5)) for _ in range(80)], 1.0, SUB, 0.3) == []
    assert g.parse_nms([[] for _ in range(80)], 1.0, SUB, 0.3) == []


def test_iou_and_overlap():
    a, b = (0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75)
    assert g.iou(a, a) == 1.0
    assert g.iou(a, b) == pytest.approx(0.0625 / (0.25 + 0.25 - 0.0625))
    assert g.iou(a, (0.6, 0.6, 0.9, 0.9)) == 0.0
    assert g.boxes_overlap(a, b) and not g.boxes_overlap(a, (0.5, 0.5, 1.0, 1.0))


def test_point_in_ellipse_respects_the_angle():
    e = (0.5, 0.5, 0.3, 0.1, 90.0)  # a tall ellipse once turned
    assert g.point_in_ellipse(0.5, 0.75, e)
    assert not g.point_in_ellipse(0.75, 0.5, e)
    assert not g.point_in_ellipse(0.5, 0.5, (0.5, 0.5, 0.0, 0.1))  # degenerate: nothing is inside


def test_point_in_polygon():
    square = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))
    assert g.point_in_polygon(0.5, 0.5, square)
    assert not g.point_in_polygon(0.1, 0.5, square)


ZONES = zones.Zones(ring=(0.45, 0.55, 0.17, 0.12, 0.13, 0.1), crop=(0.24, 0.23, 0.45, 0.5),
                    ground=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))


@pytest.mark.parametrize("box, zone", [
    ((0.43, 0.42, 0.47, 0.46), g.BATH),        # perched on the far rim
    ((0.26, 0.25, 0.30, 0.30), g.CROP),        # in the crop box but off the bath
    ((0.85, 0.85, 0.95, 0.95), g.GROUND),      # on the lawn
    ((0.9, 0.0, 1.0, 0.05), g.GROUND),         # the polygon here is the whole frame
])
def test_zone_of(box, zone):
    assert g.zone_of(box, ZONES) == zone


def test_zone_of_without_a_ring_uses_the_roi_box_and_elsewhere_is_the_default():
    z = zones.Zones(roi=(0.4, 0.4, 0.2, 0.2))
    assert g.zone_of((0.45, 0.45, 0.5, 0.5), z) == g.BATH
    assert g.zone_of((0.0, 0.0, 0.1, 0.1), z) == g.ELSEWHERE


def test_a_box_around_the_whole_bath_counts_as_the_bath():
    assert g.zone_of((0.2, 0.3, 0.7, 0.8), ZONES) == g.BATH  # the ellipse's centre is inside the box


def test_to_frame_maps_a_cut_out_detection_back_onto_the_frame():
    d = g.Detection("bird", 0.7, (0.0, 0.5, 1.0, 1.0))  # the lower half of the cut-out
    assert g.to_frame(d, (0.2, 0.2, 0.4, 0.5)).box == (0.2, 0.45, 0.6, 0.7)
    assert g.to_frame(d, None) is d


def test_frame_is_night_on_a_grey_frame():
    assert g.frame_is_night(np.full(SUB, 90, np.uint8))
    colour = np.zeros(SUB, np.uint8)
    colour[:] = (30, 200, 60)
    assert not g.frame_is_night(colour)


# --- tracker ----------------------------------------------------------------------------------------------------


def det(x0, y0, size=0.05, label="bird", score=0.8):
    return g.Detection(label, score, (x0, y0, x0 + size, y0 + size))


def test_tracker_confirms_after_min_hits_and_ends_after_the_gap():
    tr = g.Tracker(min_hits=3, gap=2.0)
    started, ended = tr.update([det(0.4, 0.4)], 0.0, ZONES)
    assert started == [] and ended == []
    started, _ = tr.update([det(0.41, 0.4)], 0.1, ZONES)
    assert started == []
    started, _ = tr.update([det(0.42, 0.41, score=0.95)], 0.2, ZONES)
    assert len(started) == 1
    track = started[0]
    assert track.hits == 3 and track.best.score == 0.95 and track.label == "bird" and track.zone == g.BATH
    assert track.animal and track.first_t == 0.0
    _, ended = tr.update([], 1.0, ZONES)
    assert ended == [] and tr.tracks == [track]
    _, ended = tr.update([], 2.3, ZONES)
    assert ended == [track] and tr.tracks == []
    assert track.seconds == pytest.approx(0.2)


def test_tracker_drops_flickers_without_reporting_them():
    tr = g.Tracker(min_hits=3, gap=1.0)
    tr.update([det(0.1, 0.1)], 0.0)
    _, ended = tr.update([], 5.0)
    assert ended == [] and tr.tracks == []


def test_tracker_follows_a_small_jump_without_overlap_and_counts_labels():
    tr = g.Tracker(min_hits=2, gap=1.0, max_jump=0.06)
    tr.update([det(0.10, 0.10, size=0.02)], 0.0)
    started, _ = tr.update([det(0.14, 0.10, size=0.02, label="cat")], 0.1)  # no overlap, 4% away
    assert len(started) == 1 and started[0].hits == 2
    assert started[0].labels == {"bird": 1, "cat": 1}


def test_tracker_keeps_two_animals_apart_and_counts_the_crowd():
    tr = g.Tracker(min_hits=2, gap=1.0)
    for t in (0.0, 0.1):
        started, _ = tr.update([det(0.1, 0.1), det(0.7, 0.7)], t)
    assert len(started) == 2
    assert all(s.concurrent == 2 for s in started)
    assert tr.next_id == 3


def test_tracker_flush_returns_only_confirmed_tracks():
    tr = g.Tracker(min_hits=2, gap=1.0)
    tr.update([det(0.1, 0.1), det(0.7, 0.7)], 0.0)
    tr.update([det(0.1, 0.1)], 0.1)
    flushed = tr.flush()
    assert [t.hits for t in flushed] == [2] and tr.tracks == []
