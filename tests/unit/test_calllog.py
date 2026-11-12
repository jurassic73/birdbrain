"""The Claude call log (birdbrain.calllog), without a database: the row built for a call and the wrapper's behaviour."""
import pytest

from helpers.claude import FakeClient, FakeResponse
from birdbrain import calllog
from birdbrain.identify import HAIKU, SONNET


@pytest.fixture
def saved(monkeypatch):
    rows = []
    monkeypatch.setattr(calllog, "save", rows.append)
    return rows


def test_row_for_a_haiku_answer_has_tokens_and_an_estimated_cost(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdef1234")
    row = calllog.row_for("watch.visit", HAIKU, 1.234, FakeResponse("end_turn", cache_read_input_tokens=800))
    assert (row["caller"], row["model"], row["ok"], row["seconds"]) == ("watch.visit", HAIKU, 1, 1.23)
    assert (row["input_tokens"], row["output_tokens"], row["cache_read_tokens"]) == (1200, 40, 800)
    assert row["usd"] == pytest.approx((1200 * 0.10 + 40 * 0.50 + 800 * 0.01) / 1e6)
    assert row["key_tail"] == "1234" and "sk-ant" not in str(row)


def test_row_records_the_model_that_answered_not_the_one_asked():
    row = calllog.row_for("x.y", SONNET, 1, FakeResponse("end_turn", model="claude-opus-5-5"))
    assert (row["requested_model"], row["model"], row["usd"]) == (SONNET, "claude-opus-5-5", None)  # no listed price


def test_row_for_an_error_keeps_one_short_line_and_no_cost():
    row = calllog.row_for("x.y", SONNET, 2, error=TimeoutError("took too long\nsecond line"))
    assert (row["ok"], row["error"], row["usd"], row["input_tokens"]) == (0, "TimeoutError: took too long", None, None)


def test_the_wrapper_logs_both_parse_paths_and_returns_the_answer(saved):
    fake = FakeClient(FakeResponse("end_turn"))
    wrapped = calllog.Logged(fake)
    assert wrapped.messages.parse(model=HAIKU) is fake.response
    assert wrapped.beta.messages.parse(model=SONNET) is fake.response
    assert [r["requested_model"] for r in saved] == [HAIKU, SONNET]
    assert saved[0]["caller"].endswith("test_the_wrapper_logs_both_parse_paths_and_returns_the_answer")


def test_the_wrapper_logs_a_failed_call_and_still_raises(saved):
    wrapped = calllog.Logged(FakeClient(raises=RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        wrapped.messages.parse(model=HAIKU)
    assert len(saved) == 1 and saved[0]["ok"] == 0 and saved[0]["error"] == "RuntimeError: boom"


def test_a_utc_day_is_shifted_to_the_local_timestamps_stored(monkeypatch):
    lo, hi = calllog.bounds("2026-11-11")
    assert (lo, hi) == ("2026-11-11T00:00:00", "2026-11-12T00:00:00")
    lo_utc, hi_utc = calllog.bounds("2026-11-11", utc=True)
    assert hi_utc > lo_utc and lo_utc[:10] in ("2026-11-10", "2026-11-11")
