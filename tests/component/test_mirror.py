"""Component tests of birdbrain/mirror.py: one real local dependency, a temp database or files on disk."""

import json
import sqlite3
from pathlib import Path

import pytest

from helpers.mirror import CHECKED_RAW
from birdbrain import config, db, mirror, privacy, zones


@pytest.fixture
def pi_db(conn, add_visit, tmp_path):
    """A Pi database with the owner's private things in it, and a mirror copy of it."""
    db.save_settings(conn, location="Vancouver, WA", scene_notes="a bath by the fence")
    for key, value in (("ntfy_topic", "secret-topic"), ("camera_log", "[]"), ("zones_request", "2026-11-03T10:00"),
                       ("pi_check_times", "{}"), ("site_timezone", "Europe/Paris"), ("species_photo", "{}"),
                       ("presence_mode", "gate"), ("zones", '{"zones": {}}'), ("zones_auto", "1"),
                       ("retention.budget_mb", "5000"), ("refphoto:Bushtit", '{"at": "2026-11-01T00:00:00"}')):
        db.set_value(conn, key, value)
    raw = {"answer": {"species": "Bushtit"}, "calls": {"haiku": {"input_tokens": 10}}, "trigger": "rim",
           "burst": {"best": 0}, "zones": {"ring": None}, "motion": {"score": 0.3}, "probe": False, "frames": 3}
    vid = add_visit(feedback="No bird: rocks after a refill", raw_json=json.dumps(raw))
    conn.execute("UPDATE visits SET public_ok = ?, clip_path = '2026/11/03/a.mp4' WHERE id = ?",
                 ('{"at": "x", "files": {}}', vid))
    conn.execute("INSERT INTO pi_checks (at, seconds, ok) VALUES ('2026-11-03T10:00:00', 4.2, 1)")
    conn.execute("INSERT INTO embeddings (visit_id, vec) VALUES (?, X'0000')", (vid,))
    db.add_calibration(conn, ran_at="2026-11-01T06:00:00", reason="scheduled", status="moved", summary="moved 3 px",
                       before_json="{}", after_json="{}", image_path="calibration/x.jpg")
    conn.commit()
    copy = tmp_path / "mirror.db"
    mirror.copy_db(config.DB_PATH, copy, "America/Los_Angeles")
    return vid, copy


@pytest.mark.smoke
def test_copy_keeps_only_the_settings_the_public_pages_read(pi_db):
    _, copy = pi_db
    keys = {k for (k,) in sqlite3.connect(copy).execute("SELECT key FROM settings")}
    assert keys == {"site_timezone", "species_photo", "presence_mode", "zones", "zones_auto", "retention.budget_mb",
                    "refphoto:Bushtit"}
    for private in ("location", "scene_notes", "ntfy_topic", "camera_log", "zones_request", "pi_check_times"):
        assert private not in keys
    # The Pi's time zone travels with the copy.
    assert sqlite3.connect(copy).execute("SELECT value FROM settings WHERE key = 'site_timezone'").fetchone()[0] == \
        "America/Los_Angeles"


def test_copy_strips_private_fields_and_tables_but_keeps_what_pages_show(pi_db, conn):
    vid, copy = pi_db
    c = sqlite3.connect(copy)
    row = c.execute("SELECT feedback, raw_json, species, notes FROM visits WHERE id = ?", (vid,)).fetchone()
    assert row[0] is None and row[2] == "Dark-eyed Junco"
    # The copy keeps the clip's path; the privacy filter (run next by prepare) blanks it unless the clip is clean.
    assert c.execute("SELECT clip_path FROM visits WHERE id = ?", (vid,)).fetchone()[0] == "2026/11/03/a.mp4"
    assert set(json.loads(row[1])) == {"calls", "trigger", "burst", "zones"}
    assert c.execute("SELECT COUNT(*) FROM pi_checks").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    assert c.execute("SELECT summary, image_path, after_json FROM calibrations").fetchone() == ("", None, "{}")
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    # The Pi's own database is untouched.
    assert db.get_visit(conn, vid)["feedback"] == "No bird: rocks after a refill"
    assert db.get_value(conn, "ntfy_topic") == "secret-topic"


@pytest.mark.smoke
def test_copy_reads_an_old_database_without_the_newer_columns(conn, add_visit, tmp_path):
    add_visit()
    old = tmp_path / "old.db"
    src, dst = sqlite3.connect(config.DB_PATH), sqlite3.connect(old)
    src.backup(dst)
    dst.execute("ALTER TABLE visits DROP COLUMN feedback")
    dst.execute("DROP TABLE pi_checks")
    dst.commit()
    dst.close()
    mirror.copy_db(old, tmp_path / "copy.db")
    assert sqlite3.connect(tmp_path / "copy.db").execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 1


@pytest.fixture
def photo_tree(tmp_path, monkeypatch, conn, add_visit):
    """An images folder with more in it than the database refers to."""
    images = tmp_path / "images"
    monkeypatch.setattr(config, "IMAGES_DIR", images)
    day = images / "2026" / "11" / "03"
    day.mkdir(parents=True)
    for name in ("a.jpg", "a_t.jpg", "a_b1.jpg", "a_b2.jpg", "a_b2_t.jpg", "a_b10.jpg", "orphan.jpg", "orphan_t.jpg",
                 ".a.jpg.thin.tmp", "a_t.jpg.tmp", "a.jpg.tmp", "gone_t.jpg", "thumb_only_t.jpg"):
        (day / name).write_bytes(b"x")
    (day / "sub").mkdir()
    add_visit(image_path="2026/11/03/a.jpg", thumb_path="2026/11/03/a_t.jpg")
    add_visit(image_path="2026/11/03/gone.jpg", thumb_path="2026/11/03/gone_t.jpg")  # the photo budget kept the thumb
    add_visit(image_path="2026/11/03/missing.jpg", thumb_path="2026/11/03/missing_t.jpg")  # files deleted
    add_visit(image_path="2026/11/04/later.jpg", thumb_path="2026/11/04/later_t.jpg")  # a folder that isn't there
    # Rows that point outside the images folder (never written by the watcher) must not be followed.
    add_visit(image_path="../../etc/passwd", thumb_path="2026/11/03/orphan_t.jpg")
    add_visit(image_path="/etc/hostname", thumb_path="/etc/hosts")
    add_visit(image_path="2026/11/03/orphan.jpg", thumb_path="2026/11/03/../../../../etc/passwd")
    return images


def test_file_list_has_only_what_the_copy_refers_to_and_never_temp_files(photo_tree):
    files = mirror.files_of(config.DB_PATH)
    # a_b2_t.jpg: the owner picked burst frame 2 as the photo, so its thumbnail goes too (images.pick_thumb)
    assert files == ["2026/11/03/a.jpg", "2026/11/03/a_b1.jpg", "2026/11/03/a_b2.jpg", "2026/11/03/a_b2_t.jpg",
                     "2026/11/03/a_b10.jpg", "2026/11/03/a_t.jpg", "2026/11/03/gone_t.jpg"]
    assert not any(f.endswith(".tmp") or "orphan" in f or "thumb_only" in f or "etc" in f for f in files)
    assert mirror.inside("2026/11/03/a.jpg") and not mirror.inside("/a.jpg") and not mirror.inside("x/../../a.jpg")
    assert not mirror.inside("")


def test_staged_links_point_at_the_listed_files_only(photo_tree, tmp_path):
    stage = tmp_path / "stage"
    files = mirror.files_of(config.DB_PATH)
    root = mirror.stage_files(files, stage)
    assert root == stage / "images"
    staged = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_symlink())
    assert staged == sorted(files)
    assert all((root / f).resolve() == (photo_tree / f).resolve() for f in files)
    assert (root / "2026/11/03/a.jpg").read_bytes() == b"x"


def test_prepare_stages_everything_the_push_sends(photo_tree, tmp_path, monkeypatch):
    class NobodyThere:
        def detect(self, view, min_score):
            return []

    import numpy as np, cv2
    for rel in ("2026/11/03/a.jpg", "2026/11/03/a_b1.jpg", "2026/11/03/a_b2.jpg", "2026/11/03/a_b10.jpg",
                "2026/11/03/gone_t.jpg"):
        cv2.imwrite(str(photo_tree / rel), np.zeros((20, 30, 3), np.uint8))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(privacy, "_det", NobodyThere())
    stage = tmp_path / "stage"
    assert mirror.main(["prepare", str(config.DB_PATH), str(stage), "Europe/Paris"]) == 0
    assert (stage / "birdbrain.db").exists() and (stage / "images" / "2026/11/03/a.jpg").is_symlink()
    listed = (stage / "files.txt").read_text().split()
    assert listed == mirror.files_of(stage / "birdbrain.db") and "2026/11/03/a_t.jpg" in listed
    assert sqlite3.connect(stage / "birdbrain.db").execute(
        "SELECT value FROM settings WHERE key = 'site_timezone'").fetchone()[0] == "Europe/Paris"


def test_prepare_fails_closed(tmp_path, capsys):
    assert mirror.main(["prepare", str(tmp_path / "nope.db"), str(tmp_path / "stage")]) == 2
    assert "mirror copy failed" in capsys.readouterr().err
    assert mirror.main(["bogus"]) == 2


def _mirror_calls(copy, vid) -> dict:
    raw = sqlite3.connect(copy).execute("SELECT raw_json FROM visits WHERE id = ?", (vid,)).fetchone()[0]
    return json.loads(raw)["calls"]


def test_circle_checks_go_to_the_mirror_by_default(conn, add_visit, tmp_path):
    vids = [add_visit(raw_json=json.dumps(CHECKED_RAW)) for _ in range(2)]
    assert mirror.owner_checks_shared(conn)  # no setting stored yet
    copy = tmp_path / "mirror.db"
    mirror.copy_db(config.DB_PATH, copy)
    for vid in vids:
        assert _mirror_calls(copy, vid)["owner_checks"][0]["say"] == "Haiku says Bushtit (high)."


def test_circle_checks_stay_on_the_pi_when_switched_off(conn, add_visit, tmp_path):
    vids = [add_visit(raw_json=json.dumps(CHECKED_RAW)) for _ in range(3)]
    db.set_value(conn, mirror.OWNER_CHECKS_KEY, "0")
    assert not mirror.owner_checks_shared(conn)
    copy = tmp_path / "mirror.db"
    mirror.copy_db(config.DB_PATH, copy)
    c = sqlite3.connect(copy)
    for vid in vids:
        calls = _mirror_calls(copy, vid)
        assert "owner_checks" not in calls
        assert calls["haiku"] == {"input_tokens": 10} and calls["circle"] == {"x": 0.5, "y": 0.5, "r": 0.1}  # Claude's stay
    assert c.execute("SELECT COUNT(*) FROM visits WHERE raw_json LIKE '%owner_checks%'").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM settings WHERE key = ?", (mirror.OWNER_CHECKS_KEY,)).fetchone()[0] == 0
    # The Pi's own database keeps the checks.
    assert json.loads(db.get_visit(conn, vids[0])["raw_json"])["calls"]["owner_checks"]
    # Switched back on, the next build carries them again: nothing is remembered from the last copy.
    db.set_value(conn, mirror.OWNER_CHECKS_KEY, "1")
    mirror.copy_db(config.DB_PATH, copy)
    assert _mirror_calls(copy, vids[0])["owner_checks"]


def test_mirror_visit_page_shows_the_checks_only_when_shared(conn, add_visit, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from birdbrain.web import app as webapp

    vid = add_visit(raw_json=json.dumps(CHECKED_RAW))
    db.set_value(conn, mirror.OWNER_CHECKS_KEY, "0")
    off, on = tmp_path / "off.db", tmp_path / "on.db"
    mirror.copy_db(config.DB_PATH, off)
    db.set_value(conn, mirror.OWNER_CHECKS_KEY, "1")
    mirror.copy_db(config.DB_PATH, on)
    monkeypatch.setenv("BIRDBRAIN_PUBLIC", "1")
    for copy, shown in ((off, False), (on, True)):
        monkeypatch.setattr(config, "DB_PATH", copy)
        with TestClient(webapp.app) as c:
            page = c.get(f"/visit/{vid}")
        assert page.status_code == 200 and "Dark-eyed Junco" in page.text
        assert ("Your circle checks" in page.text) is shown
        assert ("Haiku says Bushtit" in page.text) is shown


def test_settings_saves_the_mirror_toggle(conn):
    from fastapi.testclient import TestClient
    from birdbrain.web import app as webapp

    with TestClient(webapp.app) as c:
        page = c.get("/settings")
        assert 'id="mirror"' in page.text and 'name="owner_checks" value="1" checked' in page.text
        r = c.post("/settings/mirror", data={}, follow_redirects=False)  # box unticked
        assert r.status_code == 303 and r.headers["location"] == "/settings?mirror_saved=1#mirror"
        assert db.get_value(conn, mirror.OWNER_CHECKS_KEY) == "0" and not mirror.owner_checks_shared(conn)
        assert 'name="owner_checks" value="1" checked' not in c.get("/settings").text
        c.post("/settings/mirror", data={"owner_checks": "1"}, follow_redirects=False)
        assert mirror.owner_checks_shared(conn)
        assert 'name="owner_checks" value="1" checked' in c.get("/settings").text


def test_copy_drops_false_triggers_and_keeps_real_visits(conn, add_visit, tmp_path):
    real = add_visit(visitor="bird", species="Bushtit")
    empty = add_visit(visitor="none", species=None, image_path="2026/11/03/ft.jpg", thumb_path="2026/11/03/ft_t.jpg")
    # corrected to an animal by the owner: a real visit now, whatever Claude first said
    fixed = add_visit(visitor="none", species=None, corrected_visitor="bird", corrected_species="Junco")
    # an older row corrected before corrected_visitor existed only set corrected_species
    old_fix = add_visit(visitor="none", species=None, corrected_species="Bushtit")
    copy = tmp_path / "mirror.db"
    mirror.copy_db(config.DB_PATH, copy)
    kept = [r[0] for r in sqlite3.connect(copy).execute("SELECT id FROM visits ORDER BY id")]
    assert kept == [real, fixed, old_fix] and empty not in kept
    assert db.get_visit(conn, empty) is not None  # the Pi's own database is untouched


def test_file_list_has_no_false_trigger_photo_thumbnail_or_burst(photo_tree, tmp_path, add_visit):
    day = photo_tree / "2026" / "11" / "03"
    for name in ("ft.jpg", "ft_t.jpg", "ft_b1.jpg", "ft_b2.jpg", "ft_b3.jpg"):
        (day / name).write_bytes(b"x")
    add_visit(visitor="none", species=None, image_path="2026/11/03/ft.jpg", thumb_path="2026/11/03/ft_t.jpg")
    # even a copy that was never trimmed lists none of them
    assert not any("ft" in f for f in mirror.files_of(config.DB_PATH))
    copy = tmp_path / "mirror.db"
    mirror.copy_db(config.DB_PATH, copy)
    files = mirror.files_of(copy)
    assert not any("ft" in Path(f).name for f in files)
    assert "2026/11/03/a.jpg" in files and "2026/11/03/a_b1.jpg" in files  # real visits keep theirs
    stage = mirror.stage_files(files, tmp_path / "stage")
    assert not any("ft" in p.name for p in stage.rglob("*"))


def test_copy_keeps_nothing_trim_removed_in_its_free_pages(conn, add_visit, tmp_path):
    """trim deletes rows and blanks columns; the copy is rebuilt (VACUUM) so their bytes aren't left in free pages
    for anyone with the file to read back. The admin's own marks (a lesson switched off) go back to the default."""
    add_visit(visitor="none", species=None, image_path="2026/11/03/ft.jpg", thumb_path="2026/11/03/ft_t.jpg",
              notes="FALSE-TRIGGER-NOTES " * 20)
    vid = add_visit(feedback="OWNER-NOTE-FOR-CLAUDE " * 20, lesson_off=1)
    copy = tmp_path / "mirror.db"
    mirror.copy_db(config.DB_PATH, copy)
    data = copy.read_bytes()
    assert b"FALSE-TRIGGER-NOTES" not in data and b"OWNER-NOTE-FOR-CLAUDE" not in data
    c = sqlite3.connect(copy)
    assert c.execute("SELECT lesson_off FROM visits WHERE id = ?", (vid,)).fetchone() == (0,)
    assert c.execute("PRAGMA freelist_count").fetchone() == (0,)
    assert db.get_visit(conn, vid)["lesson_off"] == 1  # the Pi's own database is untouched


@pytest.mark.smoke
def test_copy_carries_the_zones_in_force_so_the_mirror_draws_them(conn, tmp_path, monkeypatch):
    ring = (0.5, 0.5, 0.13, 0.15, 0.11, 0.11, 0.5)
    monkeypatch.setattr(config, "MOTION_RING", ring)  # from .env on the Pi, nothing saved in the database
    copy = tmp_path / "copy.db"
    mirror.copy_db(config.DB_PATH, copy)
    monkeypatch.setattr(config, "MOTION_RING", None)  # the mirror's own .env has no zones
    pub = sqlite3.connect(copy)
    pub.row_factory = sqlite3.Row
    try:
        assert tuple(zones.load(pub).ring) == ring
        assert zones.stored(pub) is not None and zones.adjusted(pub) is False
    finally:
        pub.close()
