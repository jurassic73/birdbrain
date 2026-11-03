"""The photo budget (birdbrain.retention): thinning older real visits and keeping every photo under one limit."""
import io
import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from birdbrain import config, db, images, retention
from birdbrain.web import app as webapp

NOW = datetime(2027, 3, 1, 12, 0)  # old visits below are from 2026-11, well past the 90 full-size days
_FRAME: list[bytes] = []
_COUNT: list[int] = []


def frame_jpeg() -> bytes:
    """A camera-sized frame with real detail (made once), so photos have realistic sizes and thin for real."""
    if not _FRAME:
        # Noise made at a quarter size and enlarged: detailed like mulch, but quick to save in every test.
        img = Image.effect_noise((480, 270), 60).convert("RGB").resize((1920, 1080), Image.Resampling.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        _FRAME.append(buf.getvalue())
    return _FRAME[0]


def quick_store(when: datetime) -> tuple[str, str]:
    """images.store_frame's files without its re-encoding (the same frame every time), to keep the tests quick."""
    if len(_FRAME) < 2:
        thumb = images.make_thumb(Image.open(io.BytesIO(frame_jpeg())))
        buf = io.BytesIO()
        thumb.save(buf, "JPEG", quality=images.THUMB_QUALITY)
        _FRAME.append(buf.getvalue())
    _COUNT.append(1)
    stem = f"{when:%Y/%m/%d}/{when:%H%M%S}_{len(_COUNT):06d}"
    (config.IMAGES_DIR / stem).parent.mkdir(parents=True, exist_ok=True)
    (config.IMAGES_DIR / f"{stem}.jpg").write_bytes(frame_jpeg())
    (config.IMAGES_DIR / f"{stem}_t.jpg").write_bytes(_FRAME[1])
    return f"{stem}.jpg", f"{stem}_t.jpg"


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    """No real disk check, no phone alerts, no cached usage from another test."""
    retention.forget()
    monkeypatch.setattr(retention, "disk_free", lambda: 10**13)
    sent = []
    monkeypatch.setattr(retention.alert, "send", lambda topic, title, message, image, click: sent.append(title))
    yield sent
    retention.forget()


@pytest.fixture
def photo(add_visit, tmp_path, monkeypatch, conn):
    """Store a visit with real photo files. Rare-species protection is off unless a test turns it on."""
    monkeypatch.setattr(config, "IMAGES_DIR", tmp_path / "images")
    retention.save_photo_options(conn, rare_under=0)

    def add(when: str, species: str | None = "Dark-eyed Junco", visitor: str = "bird", **fields) -> int:
        image_rel, thumb_rel = quick_store(datetime.fromisoformat(when))
        return add_visit(captured_at=when, image_path=image_rel, thumb_path=thumb_rel, visitor=visitor,
                         species=species, **fields)

    return add


def false_trigger(photo, when: str, **fields) -> int:
    return photo(when, species=None, visitor="none", **fields)


def level(conn, vid: int) -> str | None:
    return db.get_visit(conn, vid)["photo_level"]


def size_of(vid: int, conn) -> tuple[int, int]:
    return Image.open(config.IMAGES_DIR / db.get_visit(conn, vid)["image_path"]).size


# --- options ---

def test_options_come_from_settings_then_env_then_built_in(conn, monkeypatch):
    monkeypatch.setattr(config, "PHOTO_BUDGET_MB", 2048.0)
    monkeypatch.setattr(config, "PHOTO_THIN_EDGE", 777)  # not a choice: the built-in 1280 is used
    o = retention.options(conn)
    assert (o["budget_mb"], o["thin_edge"], o["full_days"], o["full_per_day"]) == (2048.0, 1280, 90, 2)
    assert (o["warn_pct"], o["rare_under"], o["disk_floor_mb"], o["thin_on"]) == (80.0, 10, 5000.0, True)
    o = retention.save_photo_options(conn, budget_mb="3000", thin_edge=960, thin_on="off", full_per_day=3)
    assert (o["budget_mb"], o["thin_edge"], o["thin_on"], o["full_per_day"]) == (3000.0, 960, False, 3)
    assert db.get_value(conn, "retention.thin_on") == "0"


@pytest.mark.parametrize("bad", [{"thin_edge": 1000}, {"budget_mb": -1}, {"warn_pct": 120}, {"full_days": "2.5"},
                                 {"thin_on": "maybe"}, {"nonsense": 1}, {"rare_under": -3}, {"budget_mb": "nan"}])
def test_bad_options_are_refused_and_nothing_is_stored(conn, bad):
    with pytest.raises(ValueError):
        retention.save_photo_options(conn, **{"full_days": 30, **bad})
    assert retention.options(conn)["full_days"] == 90  # the good value in the same call wasn't stored either


# --- thinning one photo ---

def test_thin_photo_resizes_once_with_lanczos_quality_and_exif(tmp_path):
    img = Image.effect_noise((1920, 1080), 30).convert("RGB")
    exif = img.getexif()
    exif[0x0110] = "test camera"
    img.save(tmp_path / "a.jpg", "JPEG", quality=85, exif=exif.tobytes())
    before, after = images.thin_photo("a.jpg", 1280, images_dir=tmp_path)
    with Image.open(tmp_path / "a.jpg") as t:
        assert t.size == (1280, 720) and t.getexif().get(0x0110) == "test camera"
    assert after < before and after == (tmp_path / "a.jpg").stat().st_size
    data = (tmp_path / "a.jpg").read_bytes()
    assert images.thin_photo("a.jpg", 1280, images_dir=tmp_path) == (after, after)  # already that small: untouched
    assert (tmp_path / "a.jpg").read_bytes() == data
    assert [p.name for p in tmp_path.iterdir()] == ["a.jpg"]  # no temp file left


def test_a_crash_while_thinning_leaves_the_original_whole(tmp_path, monkeypatch):
    Image.effect_noise((1920, 1080), 30).convert("RGB").save(tmp_path / "a.jpg", "JPEG", quality=85)
    data = (tmp_path / "a.jpg").read_bytes()

    def boom(src, dst):
        raise OSError("power cut")

    monkeypatch.setattr(images.os, "replace", boom)
    with pytest.raises(OSError):
        images.thin_photo("a.jpg", 1280, images_dir=tmp_path)
    assert (tmp_path / "a.jpg").read_bytes() == data
    assert [p.name for p in tmp_path.iterdir()] == ["a.jpg"]


def test_drop_photo_keeps_the_thumbnail(tmp_path):
    rel, thumb = images.store_frame(frame_jpeg(), datetime(2026, 11, 1, 9), images_dir=tmp_path)
    images.drop_photo(rel, images_dir=tmp_path)
    assert not (tmp_path / rel).exists() and (tmp_path / thumb).exists()


# --- what's protected ---

def test_each_protection_rule(conn, photo):
    first = photo("2026-11-01T08:00:00")                                 # first Dark-eyed Junco: first sighting
    plain = photo("2026-11-01T09:00:00")
    keeper = photo("2026-11-01T10:00:00", keep=1)
    corrected = photo("2026-11-01T11:00:00", corrected_species="Song Sparrow")  # also the first Song Sparrow
    sparrow2 = photo("2026-11-02T11:00:00", species="Song Sparrow")
    noted = photo("2026-11-01T12:00:00", feedback="tail looked odd")
    circled = photo("2026-11-01T13:00:00", marks=json.dumps([{"x": 0.5, "y": 0.5, "r": 0.1}]))
    also_fixed = photo("2026-11-01T14:00:00", corrected_also="[]")
    pick = photo("2026-11-01T15:00:00")
    db.set_species_photo(conn, "Dark-eyed Junco", pick)
    ft_lesson = false_trigger(photo, "2026-11-01T16:00:00", feedback="leaf")
    p = retention.protected(conn)
    assert p[first] == "first sighting" and p[keeper] == "keeper" and p[pick] == "species photo"
    assert {p[v] for v in (corrected, noted, circled, also_fixed, ft_lesson)} == {"training"}
    assert plain not in p and sparrow2 not in p
    retention.save_photo_options(conn, rare_under=3)  # Song Sparrow has 2 visits: rare, so all of them stay
    p = retention.protected(conn)
    assert p[sparrow2] == "rare species" and plain not in p


def test_first_sighting_counts_other_animals_in_the_photo(conn, photo):
    photo("2026-11-01T08:00:00")
    with_tit = photo("2026-11-01T09:00:00", also=json.dumps([{"visitor": "bird", "species": "Bushtit", "count": 3}]))
    photo("2026-11-02T09:00:00", species="Bushtit")
    assert retention.protected(conn)[with_tit] == "first sighting"  # its first Bushtit, though a junco is the main one


# --- each species' best of the day ---

def test_best_of_day_by_sharpness_then_confidence_with_also_species(conn, photo):
    day = "2026-11-05T"
    a = photo(day + "09:00:00", photo_sharpness=50.0)
    b = photo(day + "10:00:00", photo_sharpness=900.0)
    c = photo(day + "11:00:00", photo_sharpness=400.0, confidence="low")
    d = photo(day + "12:00:00", photo_sharpness=400.0, confidence="high")
    # A blurry finch photo that's the only one with a Bushtit that day: best for the Bushtit, so it stays.
    e = photo(day + "13:00:00", species="House Finch", photo_sharpness=10.0,
              also=json.dumps([{"visitor": "bird", "species": "Bushtit", "count": 1, "confidence": "high"}]))
    other_day = photo("2026-11-06T09:00:00", photo_sharpness=1.0)
    photos = retention._photos(conn)
    best = retention.best_of_day(photos, retention._seen_rows(conn), 2)
    assert {b, d, e, other_day} <= best and a not in best and c not in best
    assert retention.best_of_day(photos, retention._seen_rows(conn), 0) == set()


def test_measure_new_scores_the_stored_photo_in_its_crop(conn, photo, monkeypatch):
    monkeypatch.setattr(config, "CROP_ROI", (0.25, 0.25, 0.5, 0.5))
    sharp = photo("2026-11-05T09:00:00")
    blurry_rel, blurry_thumb = quick_store(datetime(2026, 11, 5, 10))
    from PIL import ImageFilter
    p = config.IMAGES_DIR / blurry_rel
    Image.open(p).filter(ImageFilter.GaussianBlur(3)).save(p, "JPEG", quality=85)
    blurry = db.insert_visit(conn, captured_at="2026-11-05T10:00:00", image_path=blurry_rel, thumb_path=blurry_thumb,
                             visitor="bird", species="Dark-eyed Junco", confidence="high")
    missing = db.insert_visit(conn, captured_at="2026-11-05T11:00:00", image_path="nope.jpg", thumb_path="nope_t.jpg",
                              visitor="bird", species="Dark-eyed Junco")
    assert retention.measure_new(conn) == 3
    s = {v: db.get_visit(conn, v)["photo_sharpness"] for v in (sharp, blurry, missing)}
    assert s[sharp] > 10 * s[blurry] > 0 and s[missing] == -1
    assert retention.measure_new(conn) == 0  # measured once


# --- the daily run ---

def test_daily_thins_old_unprotected_photos_once_and_spares_the_best(conn, photo):
    retention.save_photo_options(conn, full_per_day=1)
    first = photo("2026-11-01T08:00:00")                   # first sighting
    best = photo("2026-11-02T09:00:00", photo_sharpness=900.0)
    plain = photo("2026-11-02T10:00:00", photo_sharpness=100.0)
    keeper = photo("2026-11-02T11:00:00", photo_sharpness=50.0, keep=1)
    recent = photo("2027-02-20T09:00:00", photo_sharpness=10.0)
    recent2 = photo("2027-02-20T10:00:00", photo_sharpness=20.0)
    r = retention.daily(conn, now=NOW)
    assert r["steps"][0]["count"] == 1 and r["steps"][0]["bytes"] > 0
    assert level(conn, plain) == "thinned" and size_of(plain, conn) == (1280, 720)
    assert db.get_visit(conn, plain)["photo_thinned_at"] == NOW.isoformat()
    for v in (first, best, keeper, recent, recent2):
        assert level(conn, v) is None and size_of(v, conn) == (1920, 1080)
    assert db.get_value(conn, retention.DAILY_KEY) == "2027-03-01"
    # Never shrunk again, even with a smaller size picked later.
    retention.save_photo_options(conn, thin_edge=960)
    retention.daily(conn, now=datetime(2027, 3, 2, 12))
    assert size_of(plain, conn) == (1280, 720)
    assert size_of(recent, conn) == (1920, 1080)  # still young


def test_thin_off_or_full_edge_leaves_photos_alone(conn, photo):
    photo("2026-11-01T08:00:00")
    plain = photo("2026-11-02T10:00:00")
    photo("2026-11-02T11:00:00")
    photo("2026-11-02T12:00:00")
    retention.save_photo_options(conn, full_per_day=0, thin_on="off")
    retention.daily(conn, now=NOW)
    assert level(conn, plain) is None
    retention.save_photo_options(conn, thin_on="on", thin_edge=1920)
    retention.daily(conn, now=NOW)
    assert level(conn, plain) is None


def test_daily_due_once_a_day_after_the_hour(conn):
    assert not retention.daily_due(conn, datetime(2027, 3, 1, 2, 59))
    assert retention.daily_due(conn, datetime(2027, 3, 1, 3, 0))
    assert retention.tick(conn, datetime(2027, 3, 1, 4, 0)) == "daily"
    assert retention.tick(conn, datetime(2027, 3, 1, 23, 0)) is None


# --- the budget ---

def budget_at(conn, used_bytes: int) -> None:
    retention.save_photo_options(conn, budget_mb=used_bytes / 1e6)


def test_over_budget_frees_false_triggers_first_then_thins_then_cuts_to_thumbnails(conn, photo, quiet):
    retention.save_photo_options(conn, thin_on="off", full_per_day=0)
    first = photo("2026-11-01T08:00:00")
    a = photo("2026-11-02T09:00:00")
    b = photo("2026-11-03T09:00:00")
    keeper = photo("2026-11-04T09:00:00", keep=1)
    fts = [false_trigger(photo, f"2026-11-0{d}T10:00:00") for d in (1, 2)]
    ft_keeper = false_trigger(photo, "2026-11-01T11:00:00", keep=1)
    photos = retention._photos(conn)
    used = sum(p.total for p in photos.values())
    ft_bytes = sum(photos[v].total for v in fts)
    budget_at(conn, used - ft_bytes - 1000)  # the false triggers aren't quite enough
    plan = retention.plan(conn, now=NOW)
    assert [(s["key"], s["count"]) for s in plan["steps"]] == [
        ("thin_old", 0), ("false_triggers", 2), ("thin", 1), ("thumb", 0)]
    assert level(conn, a) is None and db.get_visit(conn, fts[0])["photo_deleted_at"] is None  # a plan changes nothing
    r = retention.enforce(conn)
    assert [(s["key"], s["count"]) for s in r["steps"]] == [
        ("thin_old", 0), ("false_triggers", 2), ("thin", 1), ("thumb", 0)]
    assert all(db.get_visit(conn, v)["photo_deleted_at"] for v in fts)
    assert db.get_visit(conn, ft_keeper)["photo_deleted_at"] is None
    assert (level(conn, a), level(conn, b)) == ("thinned", None)  # the oldest unprotected one
    assert r["used_after"] <= r["budget_bytes"] and not r["still_over"] and quiet == []
    # Squeezed hard: every unprotected photo ends as a thumbnail, protected ones untouched, one alert.
    db.set_value(conn, "ntfy_topic", "test-topic")
    budget_at(conn, 1000)
    r = retention.enforce(conn)
    assert (level(conn, a), level(conn, b)) == ("thumb", "thumb")
    assert not (config.IMAGES_DIR / db.get_visit(conn, a)["image_path"]).exists()
    assert (config.IMAGES_DIR / db.get_visit(conn, a)["thumb_path"]).exists()
    for v in (first, keeper):
        assert level(conn, v) is None and size_of(v, conn) == (1920, 1080)
    assert r["still_over"] and quiet == ["Birdbrain photo storage is full"]
    retention.enforce(conn)
    assert quiet == ["Birdbrain photo storage is full"]  # not repeated
    retention.save_photo_options(conn, budget_mb=100_000)
    retention.tick(conn, datetime(2027, 3, 1, 2, 0))
    assert db.get_value(conn, retention.OVER) is None  # back under: the next squeeze alerts again


def test_budget_thins_the_days_best_last(conn, photo, tmp_path):
    retention.save_photo_options(conn, thin_on="off", full_per_day=1)
    photo("2026-11-01T08:00:00")
    best_old = photo("2026-11-02T09:00:00", photo_sharpness=900.0)
    rest_old = photo("2026-11-02T10:00:00", photo_sharpness=100.0)
    rest_new = photo("2026-11-03T10:00:00", photo_sharpness=100.0)
    photo("2026-11-03T11:00:00", photo_sharpness=900.0)
    photos = retention._photos(conn)
    used = sum(p.total for p in photos.values())
    (tmp_path / "copy").mkdir()
    (tmp_path / "copy" / "x.jpg").write_bytes((config.IMAGES_DIR / photos[rest_old].image_path).read_bytes())
    before, after = images.thin_photo("x.jpg", 1280, images_dir=tmp_path / "copy")
    budget_at(conn, used - int(1.5 * (before - after)))  # one thinned photo isn't enough, two are
    retention.enforce(conn)
    assert (level(conn, rest_old), level(conn, rest_new), level(conn, best_old)) == ("thinned", "thinned", None)


def test_disk_floor_frees_space_even_under_the_budget(conn, photo, monkeypatch):
    false_trigger(photo, "2026-11-01T10:00:00")
    ft = false_trigger(photo, "2026-11-02T10:00:00")
    retention.save_photo_options(conn, budget_mb=100_000, disk_floor_mb=1000)
    monkeypatch.setattr(retention, "disk_free", lambda: 1000 * 10**6 - 1)  # one byte short
    assert retention.tick(conn, datetime(2027, 3, 1, 2, 0)) == "enforce"
    left = [v for v in db.list_visits(conn, false_triggers=True, has_photo=True)]
    assert [v["id"] for v in left] == [ft]  # the oldest one went


def test_warning_at_the_threshold_once_until_it_drops(conn, photo, quiet):
    db.set_value(conn, "ntfy_topic", "test-topic")
    photo("2026-11-01T08:00:00")
    used = retention.photo_usage(conn)
    budget_at(conn, int(used / 0.85))
    early = datetime(2027, 3, 1, 2, 0)  # before the daily hour: a plain check
    retention.tick(conn, early)
    retention.tick(conn, early)
    assert quiet == ["Birdbrain photos are filling up"]
    budget_at(conn, int(used / 0.5))
    retention.tick(conn, early)
    assert db.get_value(conn, retention.WARNED) is None
    budget_at(conn, int(used / 0.9))
    retention.tick(conn, early)
    assert quiet == ["Birdbrain photos are filling up"] * 2


def test_no_topic_no_alert_and_it_waits_for_one(conn, photo, quiet):
    photo("2026-11-01T08:00:00")
    budget_at(conn, int(retention.photo_usage(conn) / 0.9))
    retention.tick(conn, datetime(2027, 3, 1, 2, 0))
    assert quiet == [] and db.get_value(conn, retention.WARNED) is None
    db.set_value(conn, "ntfy_topic", "test-topic")
    retention.tick(conn, datetime(2027, 3, 1, 2, 0))
    assert quiet == ["Birdbrain photos are filling up"]


def test_stored_visits_add_to_the_cached_usage(conn, photo):
    photo("2026-11-01T08:00:00")
    used = retention.photo_usage(conn)
    rel, thumb = quick_store(datetime(2026, 11, 2, 9))
    retention.note_stored(rel, thumb)
    assert retention.photo_usage(conn) == used + images.frame_bytes(rel, thumb)


# --- thinned and thumbnail-only visits on the site ---

def test_thumbnail_only_visits_still_count_everywhere(conn, photo, monkeypatch):
    monkeypatch.setattr(db.clock, "today", lambda: datetime(2026, 11, 3).date())
    photo("2026-11-01T08:00:00")
    cut = photo("2026-11-02T09:00:00")
    conn.execute("UPDATE visits SET photo_level = 'thumb' WHERE id = ?", (cut,))
    conn.commit()
    assert cut in [v["id"] for v in db.list_visits(conn)]
    assert db.count_visits(conn, species="Dark-eyed Junco") == 2
    assert db.species_row(conn, "Dark-eyed Junco")["visits"] == 2
    assert sum(n for _, n in db.daily_counts(conn, days=5)) == 2 and sum(db.hourly_counts(conn)) == 2
    assert db.scoreboard_total(conn) == 2
    v = db.get_visit(conn, cut)
    assert db.photo_rel(v) == v["thumb_path"]


@pytest.fixture
def client(conn):
    with TestClient(webapp.app) as c:
        yield c


def test_visit_page_shows_thinned_and_thumbnail_only_photos(conn, photo, client):
    thinned = photo("2026-11-01T08:00:00", photo_level="thinned")
    cut = photo("2026-11-02T09:00:00", photo_level="thumb")
    html = client.get(f"/visit/{thinned}").text
    v = db.get_visit(conn, thinned)
    assert f'src="/images/{v["image_path"]}"' in html and ">thinned<" in html
    html = client.get(f"/visit/{cut}").text
    v = db.get_visit(conn, cut)
    assert f'src="/images/{v["thumb_path"]}" width="1920"' in html and ">thumbnail only<" in html
    assert f'src="/images/{v["image_path"]}"' not in html
    assert 'class="badge photo-level" data-tip="Thumbnail only:' in html
    home = client.get("/").text
    assert f'data-full="/images/{v["thumb_path"]}"' in home


# --- breakdown and the CLI ---

def test_breakdown_counts_each_photo_once_in_order(conn, photo):
    pick = photo("2026-11-01T08:00:00", keep=1)     # a species photo and a keeper: counts as the species photo
    db.set_species_photo(conn, "Dark-eyed Junco", pick)
    photo("2026-11-02T08:00:00", keep=1)
    photo("2026-11-03T08:00:00", feedback="note")
    recent = datetime.now().replace(microsecond=0).isoformat()
    photo(recent)
    photo("2026-01-04T08:00:00", photo_level="thinned")  # older than 90 days
    false_trigger(photo, "2026-11-04T09:00:00")
    b = retention.breakdown(conn)
    counts = {c["key"]: c["count"] for c in b["categories"]}
    assert counts == {"species_photos": 1, "keepers": 1, "training": 1, "recent": 1, "older": 1, "false_triggers": 1}
    assert b["thumbs"]["count"] == 6
    assert b["used_bytes"] == sum(c["bytes"] for c in b["categories"]) + b["thumbs"]["bytes"]
    assert b["levels"] == {"full": 4, "thinned": 1, "thumb": 0}
    assert b["budget_bytes"] == int(retention.options(conn)["budget_mb"] * 1e6) and b["used_pct"] is not None
    assert set(b["system"]) == {"models", "database", "calibration"} and b["system"]["database"] > 0
    assert retention.breakdown(conn) is b  # cached for a minute


def test_cli_dry_run_changes_nothing(conn, photo, monkeypatch, capsys):
    retention.save_photo_options(conn, full_per_day=0)
    photo("2026-11-01T08:00:00")
    old = photo("2026-11-02T08:00:00")
    ft = false_trigger(photo, "2026-11-02T09:00:00")
    budget_at(conn, 1000)
    files = {p: p.stat().st_mtime_ns for p in config.IMAGES_DIR.rglob("*.jpg")}
    rows = [tuple(r) for r in conn.execute("SELECT * FROM visits ORDER BY id")]
    monkeypatch.setattr("sys.argv", ["retention", "--thin", "--dry-run"])
    retention.main()
    out = capsys.readouterr().out
    assert "dry run" in out and "would free" in out
    assert {p: p.stat().st_mtime_ns for p in config.IMAGES_DIR.rglob("*.jpg")} == files
    assert [tuple(r) for r in conn.execute("SELECT * FROM visits ORDER BY id")] == rows
    assert level(conn, old) is None and db.get_visit(conn, ft)["photo_deleted_at"] is None
    assert db.get_value(conn, retention.DAILY_KEY) is None


def test_cli_thin_reports_progress(conn, photo, monkeypatch, capsys):
    retention.save_photo_options(conn, full_per_day=0)
    photo("2026-11-01T08:00:00")
    old = photo("2026-11-02T08:00:00")
    monkeypatch.setattr("sys.argv", ["retention", "--thin"])
    monkeypatch.setattr(retention, "datetime", type("D", (datetime,), {"now": classmethod(lambda cls: NOW)}))
    retention.main()
    out = capsys.readouterr().out
    assert "1/1 thinned, 0 failed, freed" in out
    assert level(conn, old) == "thinned"
