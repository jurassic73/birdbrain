import io
from datetime import datetime

from PIL import Image

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


def test_stamp_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(images.config, "PHOTO_STAMP", False)
    image_rel, _ = images.store_frame(jpeg(), datetime(2026, 11, 1, 9, 33), tmp_path)
    full = Image.open(tmp_path / image_rel).convert("RGB")
    assert abs(sum(full.getpixel((60, 1080 - 68))) - sum((90, 120, 200))) < 30
