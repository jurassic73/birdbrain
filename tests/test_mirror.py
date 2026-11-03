import json
import sqlite3

import pytest

from birdbrain import config, db, mirror, privacy


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
    conn.execute("UPDATE visits SET public_ok = ? WHERE id = ?", ('{"at": "x", "files": {}}', vid))
    conn.execute("INSERT INTO pi_checks (at, seconds, ok) VALUES ('2026-11-03T10:00:00', 4.2, 1)")
    conn.execute("INSERT INTO embeddings (visit_id, vec) VALUES (?, X'0000')", (vid,))
    db.add_calibration(conn, ran_at="2026-11-01T06:00:00", reason="scheduled", status="moved", summary="moved 3 px",
                       before_json="{}", after_json="{}", image_path="calibration/x.jpg")
    conn.commit()
    copy = tmp_path / "mirror.db"
    mirror.copy_db(config.DB_PATH, copy, "America/Los_Angeles")
    return vid, copy


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
    assert set(json.loads(row[1])) == {"calls", "trigger", "burst", "zones"}
    assert c.execute("SELECT COUNT(*) FROM pi_checks").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    assert c.execute("SELECT summary, image_path, after_json FROM calibrations").fetchone() == ("", None, "{}")
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    # The Pi's own database is untouched.
    assert db.get_visit(conn, vid)["feedback"] == "No bird: rocks after a refill"
    assert db.get_value(conn, "ntfy_topic") == "secret-topic"


def test_public_raw_drops_what_nothing_public_reads():
    assert mirror.public_raw(None) is None and mirror.public_raw("not json") is None and mirror.public_raw("[1]") is None
    assert mirror.public_raw('{"motion": 1}') is None
    assert json.loads(mirror.public_raw('{"calls": {"x": 1}, "answer": {}, "stay_end": "t"}')) == {"calls": {"x": 1}}


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
