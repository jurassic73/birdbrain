"""The offline scorecard's pure parts: labels, views and the scorecard itself."""
import json

import numpy as np

from birdbrain import zones
from birdbrain.hailo import evaluate, geometry


class Row(dict):
    """sqlite3.Row stand-in: subscriptable by column name."""


def test_final_label_and_real_honour_corrections():
    assert evaluate.final_label(Row(visitor="bird", corrected_visitor=None)) == "bird"
    assert evaluate.final_label(Row(visitor="bird", corrected_visitor="squirrel")) == "squirrel"
    assert evaluate.is_real(Row(visitor="bird", corrected_visitor=None, corrected_species=None))
    assert not evaluate.is_real(Row(visitor="none", corrected_visitor=None, corrected_species=None))
    assert evaluate.is_real(Row(visitor="none", corrected_visitor=None, corrected_species="House Finch"))
    assert not evaluate.is_real(Row(visitor="bird", corrected_visitor="none", corrected_species=None))


def test_visit_crop_prefers_the_zones_stored_with_the_visit():
    fallback = zones.Zones(crop=(0.1, 0.1, 0.5, 0.5))
    stored = Row(raw_json=json.dumps({"zones": {"crop": [0.2, 0.2, 0.4, 0.4]}}))
    assert evaluate.visit_crop(stored, fallback) == (0.2, 0.2, 0.4, 0.4)
    assert evaluate.visit_crop(Row(raw_json=None), fallback) == (0.1, 0.1, 0.5, 0.5)
    assert evaluate.visit_crop(Row(raw_json="{not json"), fallback) == (0.1, 0.1, 0.5, 0.5)


def test_views_shrink_to_the_substream_width_and_cut_the_crop():
    img = np.zeros((1080, 1920, 3), np.uint8)
    v = evaluate.views(img, (0.25, 0.25, 0.5, 0.5))
    assert v["frame"].shape == (360, 640, 3)
    assert v["crop"].shape == (540, 960, 3)
    assert v["subcrop"].shape == (180, 320, 3)
    small = evaluate.views(np.zeros((360, 640, 3), np.uint8), None)
    assert small["frame"].shape == (360, 640, 3) and "crop" not in small and "subcrop" not in small


def test_summarize_picks_the_best_animal_not_the_best_object():
    found = [geometry.Detection("bench", 0.9, (0, 0, 0.1, 0.1)), geometry.Detection("cat", 0.6, (0, 0, 0.1, 0.1)),
             geometry.Detection("bird", 0.5, (0, 0, 0.1, 0.1))]
    s = evaluate.summarize(found)
    assert s["best"] == ["cat", 0.6] and s["found"][0] == ["bench", 0.9]
    assert evaluate.summarize([])["best"] is None


def rec(i, label, best, night=False):
    real = label != "none"
    return {"id": i, "at": "2026-11-08T12:00:00", "label": label, "real": real, "night": night, "species": None,
            "size": [1920, 1080], "modes": {"frame": {"best": best, "found": [best] if best else []}}}


ROWS = [rec(1, "bird", ["bird", 0.9]), rec(2, "bird", ["bird", 0.45]), rec(3, "bird", None),
        rec(4, "squirrel", ["cat", 0.55]), rec(5, "squirrel", ["dog", 0.25]),
        rec(6, "none", None), rec(7, "none", ["bird", 0.35]), rec(8, "none", None, night=True)]


def test_scorecard_counts_recall_precision_and_classes():
    card = evaluate.scorecard(ROWS, "frame", thresholds=(0.3, 0.5))
    assert card["real"] == 5 and card["false"] == 3 and card["false_night"] == 1
    assert card["by_type"] == {"bird": 3, "squirrel": 2}
    at3, at5 = card["thresholds"]
    assert at3["flagged_real"] == 3 and at3["recall"] == 0.6
    assert at3["recall_by_type"] == {"bird": 2 / 3, "squirrel": 0.5}
    assert at3["flagged_false"] == 1 and at3["ignored_false"] == 2 and at3["precision"] == 0.75
    assert at3["ignored_false_day"] == 1 and at3["ignored_false_night"] == 1
    assert at5["flagged_real"] == 2 and at5["flagged_false"] == 0 and at5["precision"] == 1.0
    assert card["classes"]["squirrel"] == {"cat": 1, "dog": 1}
    assert card["classes"]["bird"] == {"bird": 2, "nothing": 1}
    assert card["classes"]["false triggers"] == {"nothing": 2, "bird": 1}
    assert card["scores"]["real"]["n"] == 5 and card["scores"]["false_flagged"]["n"] == 1


def test_scorecard_with_no_rows_for_a_view_has_no_divisions_by_zero():
    card = evaluate.scorecard(ROWS, "crop", thresholds=(0.3,))
    assert card["real"] == 0 and card["thresholds"][0]["recall"] is None
    assert card["thresholds"][0]["precision"] is None


def test_markdown_report_has_a_table_per_view_and_names_the_classes():
    cards = {"yolov8s_h8": {"frame": evaluate.scorecard(ROWS, "frame", thresholds=(0.3,))}}
    meta = {"generated": "2026-11-08T12:00:00", "photos": 8, "real": 5, "false": 3,
            "timing": {"yolov8s_h8": {"seconds": 12.0, "ms": 11.2}}}
    text = evaluate.markdown(cards, meta)
    assert "## yolov8s_h8" in text and "### View: frame" in text
    assert "| 0.3 | 60% | 67% | 50% | 67% | 1/2 | 1/1 | 75% |" in text
    assert "- squirrel: cat 1, dog 1" in text
    assert "11.2 ms per view" in text
    assert "—" not in text  # no em dashes in prose this repo writes
