import io
from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from conftest import image_size
from birdbrain import config, db, images, presence, retention


def noisy_jpeg() -> bytes:
    """A frame with real detail, so it has a realistic, non-trivial file size."""
    img = Image.effect_noise((640, 360), 60).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def stored(add_visit, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    base = datetime(2026, 10, 25, 9, 0)

    def add(minutes: int, **fields) -> int:
        when = base + timedelta(minutes=minutes)
        image_rel, thumb_rel = images.store_frame(noisy_jpeg(), when)
        defaults = {"visitor": "none", "species": None}
        return add_visit(captured_at=when.isoformat(), image_path=image_rel, thumb_path=thumb_rel, **{**defaults, **fields})

    return add


def test_usage_splits_deletable_from_lesson_false_triggers(conn, stored):
    stored(0)
    stored(1)
    stored(2, feedback="rock tip, not a bird")          # a lesson: kept
    stored(3, visitor="bird", species="Bushtit")         # a real visit: not a false trigger at all
    u = retention.usage(conn)
    assert (u["count"], u["lesson_count"], u["confirmed_count"]) == (2, 1, 0)
    assert u["bytes"] > 0 and u["lesson_bytes"] > 0
    assert u["limit_bytes"] == int(config.FALSE_TRIGGER_MAX_MB * 1e6)


def test_prune_deletes_oldest_first_until_under_the_limit(conn, stored):
    ids = [stored(m) for m in range(5)]
    lesson = stored(-10, corrected_visitor="none", feedback="ripples")  # oldest of all, but a lesson
    one = retention.usage(conn)["bytes"] / 5
    deleted = retention.prune(conn, limit_bytes=int(one * 2.5))
    assert deleted == 3
    left = {r["id"] for r in db.list_visits(conn, false_triggers=True, has_photo=True, limit=50)}
    assert left == {ids[3], ids[4], lesson}
    assert retention.usage(conn)["bytes"] <= one * 2.5
    assert db.count_visits(conn, false_triggers=True) == 6  # the records stay; only the photos went
    assert retention.usage(conn)["photos_deleted"] == 3


def test_prune_takes_confirmed_photos_first(conn, stored):
    ids = [stored(m) for m in range(4)]
    db.confirm_visits(conn, [ids[3]])  # the newest, but checked by the owner
    one = retention.usage(conn)["bytes"] / 4
    assert retention.prune(conn, limit_bytes=int(one * 2.5)) == 2
    left = {r["id"] for r in db.list_visits(conn, false_triggers=True, has_photo=True, limit=50)}
    assert left == {ids[1], ids[2]}


def test_deleting_photos_keeps_the_record_and_its_training(conn, stored):
    vid = stored(0)
    presence.store(conn, vid, np.ones(8))
    row = db.get_visit(conn, vid)
    retention.prune(conn, limit_bytes=1)
    after = db.get_visit(conn, vid)
    assert after and after["photo_deleted_at"]
    assert not (config.IMAGES_DIR / row["image_path"]).exists()
    assert list(presence.labelled(conn)[3]) == [vid]  # the Pi's animal check still learns from it
    assert db.count_unchecked_false_triggers(conn) == 0  # nothing left to look at


def test_prune_removes_the_image_files(conn, stored):
    vid = stored(0)
    row = db.get_visit(conn, vid)
    retention.prune(conn, limit_bytes=1)
    assert not (config.IMAGES_DIR / row["image_path"]).exists()
    assert not (config.IMAGES_DIR / row["thumb_path"]).exists()


def test_settings_page_values_override_env_and_lowering_the_limit_prunes(conn, stored, monkeypatch):
    monkeypatch.setattr(config, "FALSE_TRIGGER_PHOTO", "large")
    assert (retention.limit_mb(conn), retention.photo_size(conn)) == (config.FALSE_TRIGGER_MAX_MB, "large")
    ids = [stored(m) for m in range(4)]
    one = retention.usage(conn)["bytes"] / 4
    deleted = retention.save_options(conn, limit=one * 2.5 / 1e6, photo="small")
    assert deleted == 2 and retention.photo_size(conn) == "small"
    assert {r["id"] for r in db.list_visits(conn, false_triggers=True, has_photo=True, limit=50)} == set(ids[2:])
    assert retention.save_options(conn, limit=0, photo="full") == 0 and retention.usage(conn)["limit_bytes"] == 0
    with pytest.raises(ValueError):
        retention.save_options(conn, limit=100, photo="huge")


def test_an_unknown_photo_size_in_env_falls_back_to_medium(conn, monkeypatch):
    monkeypatch.setattr(config, "FALSE_TRIGGER_PHOTO", "tiny")
    assert retention.photo_size(conn) == "medium"


def test_zero_limit_means_no_limit(conn, stored):
    stored(0)
    assert retention.prune(conn, limit_bytes=0) == 0


def test_clear_deletes_only_photos_the_owner_confirmed_empty(conn, stored):
    checked = [stored(0), stored(1)]
    unchecked = stored(2)
    lesson = stored(3, feedback="leaf", confirmed_at="2026-10-31T09:00:00")  # confirmed, but also a lesson
    real = stored(4, visitor="raccoon", confirmed_at="2026-10-31T09:00:00")
    db.confirm_visits(conn, checked)
    u = retention.usage(conn)
    assert (u["count"], u["confirmed_count"]) == (3, 2) and 0 < u["confirmed_bytes"] < u["bytes"]
    assert retention.clear(conn) == 2
    has_photo = lambda i: db.get_visit(conn, i)["photo_deleted_at"] is None
    assert [has_photo(i) for i in [*checked, unchecked, lesson, real]] == [False, False, True, True, True]
    assert retention.usage(conn)["count"] == 1


def test_false_triggers_are_saved_smaller(tmp_path):
    big = Image.effect_noise((1920, 1080), 60).convert("RGB")
    buf = io.BytesIO()
    big.save(buf, "JPEG", quality=97)
    full_rel, _ = images.store_frame(buf.getvalue(), datetime(2026, 10, 25), tmp_path)
    small_rel, _ = images.store_frame(buf.getvalue(), datetime(2026, 10, 25), tmp_path, max_edge=960, quality=80)
    assert image_size(tmp_path / full_rel) == (1920, 1080)
    assert image_size(tmp_path / small_rel) == (960, 540)
    assert (tmp_path / small_rel).stat().st_size < (tmp_path / full_rel).stat().st_size / 2
