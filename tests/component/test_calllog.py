"""The Claude call log written to and summarised from a real (temporary) database."""
from datetime import datetime

from helpers.claude import FakeClient, FakeResponse
from birdbrain import calllog, db
from birdbrain.identify import HAIKU, SONNET


def test_a_logged_call_lands_in_the_table_and_the_summary_groups_it(conn):
    wrapped = calllog.Logged(FakeClient(FakeResponse("end_turn", model=SONNET)))
    wrapped.beta.messages.parse(model=SONNET)
    wrapped.beta.messages.parse(model=SONNET)
    calllog.Logged(FakeClient(FakeResponse("end_turn", model=HAIKU))).messages.parse(model=HAIKU)
    today = datetime.now().date().isoformat()
    rows = calllog.summary(conn, today)
    assert sorted((r["model"], r["calls"]) for r in rows) == [(HAIKU, 1), (SONNET, 2)]
    assert all("test_calllog." in r["caller"] and r["failed"] == 0 for r in rows)
    assert calllog.summary(conn, "2020-01-01") == []


def test_the_table_is_part_of_the_schema(conn):
    assert "claude_calls" in {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_a_database_that_cannot_be_written_never_breaks_the_call(conn, monkeypatch):
    monkeypatch.setattr(db, "session", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    answer = calllog.Logged(FakeClient(FakeResponse("end_turn"))).messages.parse(model=HAIKU)
    assert answer.stop_reason == "end_turn"
