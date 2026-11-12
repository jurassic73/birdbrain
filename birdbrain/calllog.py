"""A log of every call to the Claude API (table claude_calls): who asked, which model, tokens, seconds, estimated cost.

identify.client() hands out a Logged wrapper, so the visit identifier, Ask Haiku/Sonnet, Refine, the lessons summary,
species names and the species fact pool are all caught in one place, on whichever machine they run (the host column
says which). Writing is best effort: a locked or missing database never breaks a call.

    uv run python -m birdbrain.calllog                     # today, local time
    uv run python -m birdbrain.calllog 2026-11-11 --utc    # a UTC day, as the Anthropic console counts it
"""
import logging
import os
import socket
import sys
import time
from datetime import datetime, time as dtime, timedelta, timezone

from . import db

log = logging.getLogger(__name__)
SKIP = {__name__, "birdbrain.identify"}  # frames that are plumbing, not the caller


def price_of(model: str | None) -> dict | None:
    name = (model or "").lower()
    return next((p for key, p in db.USD_PER_MTOK.items() if key in name), None)


def usage_of(response) -> dict:
    u = getattr(response, "usage", None)
    return {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None),
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", None),
            "cache_write_tokens": getattr(u, "cache_creation_input_tokens", None)}


def row_for(caller: str, requested: str | None, seconds: float, response=None,
            error: BaseException | None = None) -> dict:
    """The claude_calls row for one call: from its response, or from the exception when there was none."""
    usage = usage_of(response)
    model = getattr(response, "model", None) or requested
    price = price_of(model)
    usd = None
    if price and response is not None:
        usd = db.call_usd(price, {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                                  "cache_read_input_tokens": usage["cache_read_tokens"],
                                  "cache_creation_input_tokens": usage["cache_write_tokens"]})
    key = os.environ.get("ANTHROPIC_API_KEY") or ""
    return {"at": datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(), "caller": caller,
            "requested_model": requested, "model": model, "ok": int(error is None),
            "stop_reason": getattr(response, "stop_reason", None),
            "error": f"{type(error).__name__}: {error}".splitlines()[0][:300] if error else None,
            "seconds": round(seconds, 2), **usage, "usd": usd, "key_tail": key[-4:] or None}


def save(row: dict) -> None:
    try:
        with db.session() as conn:
            conn.executescript(db.CLAUDE_CALLS_SQL)  # IF NOT EXISTS: the first call may come before init_db
            conn.execute(f"INSERT INTO claude_calls ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                         list(row.values()))
            conn.commit()
    except Exception as e:  # never let bookkeeping break a call that worked
        log.warning("claude call not logged: %s: %s", type(e).__name__, e)


def caller_name() -> str:
    """module.function of the nearest frame outside this module and identify (which only forwards the request)."""
    f = sys._getframe(1)
    while f and f.f_globals.get("__name__") in SKIP:
        f = f.f_back
    if f is None:
        return "?"
    return f"{(f.f_globals.get('__name__') or '?').removeprefix('birdbrain.')}.{f.f_code.co_name}"


class _Messages:
    def __init__(self, target):
        self._target = target

    def parse(self, **kwargs):
        caller, started = caller_name(), time.monotonic()
        try:
            response = self._target.parse(**kwargs)
        except BaseException as e:
            save(row_for(caller, kwargs.get("model"), time.monotonic() - started, error=e))
            raise
        save(row_for(caller, kwargs.get("model"), time.monotonic() - started, response=response))
        return response

    def __getattr__(self, name):
        return getattr(self._target, name)


class _Beta:
    def __init__(self, beta):
        self._beta = beta
        self.messages = _Messages(beta.messages)

    def __getattr__(self, name):
        return getattr(self._beta, name)


class Logged:
    """The anthropic client with messages.parse and beta.messages.parse logged; everything else passes through."""

    def __init__(self, real):
        self._real = real
        self.messages = _Messages(real.messages)
        self.beta = _Beta(real.beta)

    def __getattr__(self, name):
        return getattr(self._real, name)


def bounds(day: str, utc: bool = False) -> tuple[str, str]:
    """[start, end) of a day as the local timestamps stored in `at`; with utc, the UTC day the console counts."""
    start = datetime.combine(datetime.fromisoformat(day).date(), dtime.min)
    if utc:
        start = start.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    return start.isoformat(timespec="seconds"), (start + timedelta(days=1)).isoformat(timespec="seconds")


def summary(conn, day: str, utc: bool = False) -> list[dict]:
    lo, hi = bounds(day, utc)
    rows = conn.execute(
        """SELECT host, caller, COALESCE(model, requested_model) AS model, COUNT(*) AS calls, SUM(1 - ok) AS failed,
                  COALESCE(SUM(usd), 0) AS usd FROM claude_calls WHERE at >= ? AND at < ?
           GROUP BY host, caller, 3 ORDER BY usd DESC, calls DESC""", (lo, hi))
    return [dict(r) for r in rows]


def main(argv: list[str]) -> None:
    utc = "--utc" in argv
    args = [a for a in argv if not a.startswith("--")]
    day = args[0] if args else datetime.now().date().isoformat()
    with db.session() as conn:
        conn.executescript(db.CLAUDE_CALLS_SQL)
        rows = summary(conn, day, utc)
    print(f"Claude calls on {day}{' (UTC)' if utc else ''}")
    for r in rows:
        print(f"{r['calls']:>5} calls {r['failed']:>2} failed  ${r['usd']:.4f}  {r['model']:<20} {r['host']:<12} "
              f"{r['caller']}")
    print(f"{sum(r['calls'] for r in rows):>5} calls total  ${sum(r['usd'] for r in rows):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
