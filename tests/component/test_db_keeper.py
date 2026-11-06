"""Component tests of birdbrain/db.py (keeper): one real local dependency, a temp database or files on disk."""

import sqlite3

from birdbrain import db


def test_init_db_adds_keep_to_an_old_database():
    old = sqlite3.connect(":memory:")
    old.row_factory = sqlite3.Row
    old.execute("CREATE TABLE visits (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL)")
    db.init_db(old)
    assert "keep" in {r["name"] for r in old.execute("PRAGMA table_info(visits)")}


def test_set_keep_toggles(conn, add_visit):
    vid = add_visit()
    assert not db.get_visit(conn, vid)["keep"]
    db.set_keep(conn, vid, True)
    assert db.get_visit(conn, vid)["keep"] == 1
    db.set_keep(conn, vid, False)
    assert not db.get_visit(conn, vid)["keep"]
