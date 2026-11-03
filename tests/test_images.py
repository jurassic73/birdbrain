import io
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from birdbrain import images


def jpeg(size=(1920, 1080)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


def test_store_frame_writes_dated_image_and_small_thumbnail(tmp_path):
    data = jpeg()
    image_rel, thumb_rel = images.store_frame(data, datetime(2026, 10, 25, 9, 26, 25), tmp_path)
    assert image_rel.startswith("2026/10/25/092625_") and image_rel.endswith(".jpg")
    assert thumb_rel.endswith("_t.jpg")
    assert Image.open(tmp_path / image_rel).size == (1920, 1080)  # re-encoded, full resolution kept
    assert images.frame_bytes(image_rel, thumb_rel, tmp_path) > 0
    assert images.frame_bytes("missing.jpg", "missing_t.jpg", tmp_path) == 0
    thumb = Image.open(tmp_path / thumb_rel)
    assert max(thumb.size) == images.THUMB_PX
    assert abs(thumb.size[0] / thumb.size[1] - 1920 / 1080) < 0.01


def test_thumbnail_is_big_and_sharp_enough_for_a_retina_grid_tile(tmp_path):
    # Fine stripes (8 px a cycle on the 1920 px frame): the old 400 px thumbnail blurred them to grey (row spread ~28)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:, (np.arange(1920) // 4) % 2 == 0] = 200
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=95)
    _, thumb_rel = images.store_frame(buf.getvalue(), datetime(2026, 11, 2, 10, 0), tmp_path)
    thumb = Image.open(tmp_path / thumb_rel)
    assert thumb.size == (800, 450)  # a 233x175 px tile is ~470x350 device px at 2x; cover from 16:9 needs ~620x350
    assert np.asarray(thumb.convert("L")).astype(float)[225].std() > 70


def old_thumb(tmp_path, stamped=False, size=(1920, 1080)):
    """A visit stored before 2026-11-02: a 400 px thumbnail of the clean frame next to the (maybe stamped) photo."""
    when = datetime(2026, 11, 1, 9, 33)
    frame = Image.new("RGB", size, (90, 120, 200))
    ImageDraw.Draw(frame).rectangle((900, 400, 1000, 500), fill=(250, 240, 30))
    image_rel, thumb_rel = "2026/11/01/093300_abcdef.jpg", "2026/11/01/093300_abcdef_t.jpg"
    (tmp_path / image_rel).parent.mkdir(parents=True)
    (images.stamp(frame, when) if stamped else frame).save(tmp_path / image_rel, quality=85)
    small = frame.copy()
    small.thumbnail((400, 400))
    small.save(tmp_path / thumb_rel, quality=82)
    return image_rel, thumb_rel, when


def test_rebuild_thumb_remakes_an_old_thumbnail_and_leaves_the_photo_and_burst_alone(tmp_path):
    image_rel, thumb_rel, when = old_thumb(tmp_path)
    burst = images.store_burst([jpeg()], image_rel, tmp_path)
    photo, frame = (tmp_path / image_rel).read_bytes(), (tmp_path / burst[0]).read_bytes()
    status, before, after = images.rebuild_thumb(image_rel, thumb_rel, when, tmp_path)
    assert status == "rebuilt" and after > before > 0
    assert Image.open(tmp_path / thumb_rel).size == (800, 450)
    assert (tmp_path / thumb_rel).stat().st_size == after
    assert (tmp_path / image_rel).read_bytes() == photo and (tmp_path / burst[0]).read_bytes() == frame
    assert sorted(p.name for p in (tmp_path / image_rel).parent.iterdir()) == sorted(
        Path(r).name for r in (image_rel, thumb_rel, *burst))  # no temp file left behind
    # Safe to run again: an up-to-date thumbnail is left as it is
    rebuilt = (tmp_path / thumb_rel).read_bytes()
    assert images.rebuild_thumb(image_rel, thumb_rel, when, tmp_path)[0] == "current"
    assert (tmp_path / thumb_rel).read_bytes() == rebuilt


def test_rebuild_thumb_of_a_small_false_trigger_photo_is_photo_sized(tmp_path):
    image_rel, thumb_rel, when = old_thumb(tmp_path, size=(640, 360))
    assert images.rebuild_thumb(image_rel, thumb_rel, when, tmp_path)[0] == "rebuilt"
    assert Image.open(tmp_path / thumb_rel).size == (640, 360)  # never enlarged
    assert images.rebuild_thumb(image_rel, thumb_rel, when, tmp_path)[0] == "current"


def test_rebuild_thumb_covers_the_time_stamp_with_the_old_thumbnails_clean_corner(tmp_path):
    image_rel, thumb_rel, when = old_thumb(tmp_path, stamped=True)
    assert images.rebuild_thumb(image_rel, thumb_rel, when, tmp_path)[0] == "stamp"
    thumb = np.asarray(Image.open(tmp_path / thumb_rel).convert("RGB")).astype(int)
    corner = thumb[-45:, :200]  # where the pill was (64 px high, ~400 wide on the full frame)
    assert np.abs(corner - (90, 120, 200)).max() < 25
    assert thumb[175:200, 380:410].mean(axis=(0, 1))[2] < 100  # the rest is the new, sharp thumbnail (yellow square)
    # An unstamped photo is not patched
    image_rel, thumb_rel, when = old_thumb(tmp_path / "clean")
    assert images.rebuild_thumb(image_rel, thumb_rel, when, tmp_path / "clean")[0] == "rebuilt"


def test_rebuild_thumb_skips_missing_files_and_dry_run_writes_nothing(tmp_path):
    image_rel, thumb_rel, when = old_thumb(tmp_path)
    before = (tmp_path / thumb_rel).read_bytes()
    status, old_bytes, new_bytes = images.rebuild_thumb(image_rel, thumb_rel, when, tmp_path, dry_run=True)
    assert status == "rebuilt" and new_bytes > old_bytes == len(before)
    assert (tmp_path / thumb_rel).read_bytes() == before
    assert len(list((tmp_path / image_rel).parent.iterdir())) == 2
    assert images.rebuild_thumb("2026/01/01/gone.jpg", "2026/01/01/gone_t.jpg", when, tmp_path) == ("missing", 0, 0)
    assert not (tmp_path / "2026/01/01").exists()
    (tmp_path / image_rel).unlink()  # photo gone, thumbnail still there: left alone
    assert images.rebuild_thumb(image_rel, thumb_rel, when, tmp_path)[0] == "missing"
    assert (tmp_path / thumb_rel).read_bytes() == before


def test_rebuild_thumbs_counts_and_reports_progress(tmp_path):
    image_rel, thumb_rel, when = old_thumb(tmp_path)
    rows = [(image_rel, thumb_rel, when.isoformat()), ("x.jpg", "x_t.jpg", when.isoformat()),
            ("bad.jpg", thumb_rel, "not a date")]
    lines = []
    counts = images.rebuild_thumbs(rows, tmp_path, log=lines.append, every=2)
    assert (counts["rebuilt"], counts["missing"], counts["failed"]) == (1, 1, 1)
    assert counts["bytes_after"] > counts["bytes_before"] > 0
    assert lines[0].startswith("2/3 done, 0 failed, ~") and lines[1].startswith("failed: ")
    assert lines[-1].startswith("3/3 done, 1 failed, ~") and lines[-1].endswith(" s left")


def test_rebuild_thumbs_command_reads_the_visits_table(tmp_path, monkeypatch, add_visit, capsys):
    image_rel, thumb_rel, when = old_thumb(tmp_path)
    add_visit(image_path=image_rel, thumb_path=thumb_rel, captured_at=when.isoformat())
    monkeypatch.setattr(images.config, "IMAGES_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["images", "--rebuild-thumbs", "--dry-run"])
    before = (tmp_path / thumb_rel).read_bytes()
    images.main()
    out = capsys.readouterr().out
    assert "1/1 done, 0 failed" in out and "would rebuild 1 (0 with the time stamp covered)" in out
    assert (tmp_path / thumb_rel).read_bytes() == before


def test_two_frames_in_the_same_second_do_not_collide(tmp_path):
    when = datetime(2026, 10, 25, 9, 0, 0)
    assert images.store_frame(jpeg(), when, tmp_path)[0] != images.store_frame(jpeg(), when, tmp_path)[0]


def test_delete_frame_removes_both_and_tolerates_missing_files(tmp_path):
    image_rel, thumb_rel = images.store_frame(jpeg(), datetime(2026, 10, 25), tmp_path)
    images.delete_frame(image_rel, thumb_rel, tmp_path)
    assert not (tmp_path / image_rel).exists() and not (tmp_path / thumb_rel).exists()
    images.delete_frame(image_rel, thumb_rel, tmp_path)  # already gone: no error


def test_burst_is_saved_in_order_counted_and_deleted_with_its_visit(tmp_path):
    image_rel, thumb_rel = images.store_frame(jpeg(), datetime(2026, 10, 30), tmp_path)
    alone = images.frame_bytes(image_rel, thumb_rel, tmp_path)
    saved = images.store_burst([jpeg()] * 11, image_rel, tmp_path, max_edge=960)
    assert images.burst_paths(image_rel, tmp_path) == saved  # _b10 and _b11 sort after _b9
    assert saved[-1].endswith("_b11.jpg") and Image.open(tmp_path / saved[0]).size == (960, 540)
    assert images.frame_bytes(image_rel, thumb_rel, tmp_path) > alone
    images.delete_frame(image_rel, thumb_rel, tmp_path)
    assert images.burst_paths(image_rel, tmp_path) == []


def test_frames_bytes_counts_each_visit_in_a_shared_folder_once(tmp_path):
    when = datetime(2026, 10, 30, 12, 0, 0)
    a = images.store_frame(jpeg(), when, tmp_path)
    b = images.store_frame(jpeg(), when, tmp_path)
    burst = images.store_burst([jpeg()] * 3, a[0], tmp_path, max_edge=640)
    size = lambda *rels: sum((tmp_path / r).stat().st_size for r in rels)
    assert images.frames_bytes([a, b, ("gone/x.jpg", "gone/x_t.jpg")], tmp_path) == [size(*a, *burst), size(*b), 0]


def test_stored_photo_gets_a_time_stamp_bottom_left_but_the_thumbnail_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(images.config, "PHOTO_STAMP", True)
    image_rel, thumb_rel = images.store_frame(jpeg(), datetime(2026, 11, 1, 9, 33, 50), tmp_path)
    full = Image.open(tmp_path / image_rel).convert("RGB")
    assert sum(full.getpixel((60, 1080 - 68))) < sum((90, 120, 200)) - 60  # the pill darkens the corner
    assert abs(sum(full.getpixel((1800, 100))) - sum((90, 120, 200))) < 30  # the rest is untouched
    thumb = Image.open(tmp_path / thumb_rel).convert("RGB")
    assert abs(sum(thumb.getpixel((12, thumb.height - 12))) - sum((90, 120, 200))) < 30


def test_stamp_reads_like_a_label():
    assert images.stamp_text(datetime(2026, 11, 1, 9, 33, 50)) == "Sun Nov 1  \u00b7  9:33 AM"
    assert images.stamp_text(datetime(2026, 10, 27, 19, 5)) == "Tue Oct 27  \u00b7  7:05 PM"
    assert images.stamp_text(datetime(2026, 10, 27, 0, 5)) == "Tue Oct 27  \u00b7  12:05 AM"


@pytest.mark.parametrize(
    "font", ["/no/such/font.ttf"] + [f for f in images.config.STAMP_FONTS if os.path.exists(f)]
)  # Pillow's own font, plus whichever of the listed fonts this machine has (the Pi's and the Mac's differ)
def test_stamp_text_and_dot_sit_on_the_middle_of_the_pill(monkeypatch, font):
    monkeypatch.setattr(images.config, "STAMP_FONTS", [font])
    for size in ((1920, 1080), (960, 540)):
        s = size[1] / 1080
        # "Sep" and "Aug" have tails below the line; the caps and digits must still be centred, not the tails
        for when in (datetime(2026, 11, 1, 14, 40), datetime(2026, 9, 6, 17, 35), datetime(2026, 8, 2, 9, 5)):
            px = np.asarray(images.stamp(Image.new("RGB", size, (60, 60, 60)), when)).astype(int)
            middle = size[1] - 36 * s - 64 * s / 2
            # the time ("2:40 PM") is all digits and capitals; measure its ink at the right end of the pill
            text = px[:, int(82 * s) :].min(2) > 160
            right = np.where(text.any(0))[0].max()
            rows = np.where(text[:, right - int(150 * s) : right + 1].any(1))[0]
            assert abs((rows.min() + rows.max() + 1) / 2 - middle) <= 1, (font, size, when, rows.min(), rows.max())
            green = (px[:, : int(82 * s), 1] - px[:, : int(82 * s), 0]) > 40
            dot = np.where(green.any(1))[0]
            assert abs((dot.min() + dot.max() + 1) / 2 - middle) <= 1


def test_stamp_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(images.config, "PHOTO_STAMP", False)
    image_rel, _ = images.store_frame(jpeg(), datetime(2026, 11, 1, 9, 33), tmp_path)
    full = Image.open(tmp_path / image_rel).convert("RGB")
    assert abs(sum(full.getpixel((60, 1080 - 68))) - sum((90, 120, 200))) < 30
