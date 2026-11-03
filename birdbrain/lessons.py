"""Lessons summary: Claude distills every correction and note the owner saved into a short list of points.

Only the newest db.LESSON_LIMIT lessons go into the identify prompt word for word, so older notes drop out. The
summary keeps what they taught: Sonnet reads all of them (every one not switched off, old ones too) and returns
the distinct, site-specific points, deduplicated. The result is always a draft; the owner reads it on Claude's
Notes (edits it if they like) and accepts or discards it. The accepted text lives in the settings table as
`lessons_summary` (part of db.SETTINGS, so db.prompt_site carries it into every prompt alongside the raw lessons).

A distill also runs on its own, in a background thread of the web process, once AUTO_EVERY new notes or
corrections have been saved since the last distill (web/app.py calls maybe_auto after each save). Failures are
logged and tried again later; nothing here can break a request or the watcher.

The SDK is imported lazily so the web app doesn't load it at startup.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime

from pydantic import BaseModel

from . import db

log = logging.getLogger(__name__)

# Sonnet, not Haiku: a distill runs about once per AUTO_EVERY notes (a few times a week at most) and costs around
# a cent, while its result rides in every identify prompt until the next one. Merging repeats and settling
# contradicting notes is where the cheaper model slips, and a wrong fold-in misleads every photo after it.
MODEL = "claude-sonnet-5-5"
MODEL_KEY = "sonnet"
AUTO_EVERY = 10          # new notes or corrections since the last distill that start one on their own
RETRY_AFTER = 15 * 60    # seconds before a failed automatic distill is tried again
MAX_POINTS = 40          # a longer list is a sign the model listed the notes instead of distilling them
BACKGROUND = True        # tests run the automatic distill in the request itself

# Settings keys. The accepted text is db.SETTINGS["lessons_summary"]; these are the bookkeeping around it.
SUMMARY = "lessons_summary"
SUMMARY_META = "lessons_summary_meta"  # JSON {at, model, usd, notes, seen, edited}: how the accepted text was made
DRAFT = "lessons_draft"                # JSON {text, changes, at, model, usd, notes, seen, auto}: waiting for the owner
DISTILLED = "lessons_distilled"        # JSON {visit_id: hash}: the lessons the last distill (draft or accepted) read
FAILED = "lessons_distill_failed"      # JSON {at, error}: the last automatic distill that didn't work

PROMPT = """You distill the notes an owner wrote about their backyard wildlife camera into a short list of points.
Another model identifies animals in each photo from this camera, and your list goes into its prompt with every
photo, so every point must be a fact about this camera, this yard or how this owner wants things named or counted
that helps that model answer better.

You get every correction and note the owner saved, oldest first. Each line says what the model answered (or that
the Pi settled the photo without asking it), what the owner says was really there, the owner's note, and sometimes
where they circled on the frame.

Rules:
- Keep every distinct fact: things here that look like animals but are not (and when: night infrared, after a
  refill, wind, rain), which species visit and how they are told apart here, the owner's naming preferences,
  counting rules, anything about day or night frames.
- One fact per point, one short plain line each. Merge repeats into one point; a fact the owner made several times
  can say so ("often mistaken for").
- Drop what is only about one photo: dates, times, circle positions, the model's exact answer. Keep the pattern they
  show.
- Never add facts that aren't in the notes. Never guess species, sizes or reasons the owner didn't give.
- When notes disagree, the later note wins; say so in changes.
- No instructions about the answer's format; that is set elsewhere.
- The whole list must have fewer words than the notes, and far fewer points than there are notes.

changes: one or two short lines for the owner on what you merged, dropped or found contradictory."""


class Distilled(BaseModel):
    points: list[str]
    changes: str


def ask(lines: list[str]) -> tuple[Distilled | None, dict]:
    """Sonnet's distillation of the lesson lines, plus token use. None when it declined."""
    from .identify import client

    text = "The owner's corrections and notes, oldest first:\n\n" + "\n".join(lines)
    response = client().messages.parse(
        model=MODEL,
        max_tokens=4000,
        output_config={"effort": "medium"},
        system=PROMPT,
        messages=[{"role": "user", "content": text}],
        output_format=Distilled,
    )
    usage = {"model": response.model, "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return None, usage
    return response.parsed_output, usage


def cost(usage: dict) -> float:
    price = db.USD_PER_MTOK[MODEL_KEY]
    return (usage.get("input_tokens", 0) * price["input"] + usage.get("output_tokens", 0) * price["output"]) / 1e6


def words(*texts: str) -> int:
    return sum(len(t.split()) for t in texts)


def _hash(line: str) -> str:
    return hashlib.sha1(line.encode()).hexdigest()[:10]


def fingerprint(lessons: list[dict]) -> dict[str, str]:
    """{visit id: hash of its line} for the lessons that would be distilled (every one not switched off)."""
    return {str(l["visit_id"]): _hash(l["line"]) for l in lessons if not l["off"]}


def new_since(conn, lessons: list[dict] | None = None) -> int:
    """How many lessons (not switched off) were saved or changed since the last distill, draft or accepted.
    With no distill yet, every lesson is new."""
    lessons = db.all_lessons(conn) if lessons is None else lessons
    seen = db.get_json(conn, DISTILLED) or {}
    return sum(1 for vid, h in fingerprint(lessons).items() if seen.get(vid) != h)


def in_summary(conn, lessons: list[dict]) -> set[int]:
    """Visit ids of the lessons folded into the accepted summary as they read now (an edited lesson isn't)."""
    meta = db.get_json(conn, SUMMARY_META) or {}
    seen = meta.get("seen") or {}
    return {int(vid) for vid, h in fingerprint(lessons).items() if seen.get(vid) == h}


def status(conn, lessons: list[dict] | None = None) -> dict:
    """Everything the Claude's Notes section shows: the accepted summary, a pending draft, counts, the last failure."""
    lessons = db.all_lessons(conn) if lessons is None else lessons
    return {
        "text": db.get_settings(conn).get(SUMMARY, ""),
        "meta": db.get_json(conn, SUMMARY_META),
        "draft": db.get_json(conn, DRAFT),
        "new": new_since(conn, lessons),
        "lessons": sum(1 for l in lessons if not l["off"]),
        "failed": db.get_json(conn, FAILED),
        "auto_every": AUTO_EVERY,
    }


def distill(conn, auto: bool = False) -> dict:
    """Ask Claude for a summary of every lesson not switched off and save the answer as the draft (never as the
    summary itself). Returns {"ok": True, "draft": ...} or {"ok": False, "error": "..."} and never raises: an
    automatic run records its failure under FAILED so it isn't retried at once."""
    lessons = [l for l in db.all_lessons(conn) if not l["off"]]
    if not lessons:
        return _failed(conn, "There are no lessons to distill yet.", auto)
    lines = [l["line"] for l in reversed(lessons)]  # all_lessons is newest first; Claude reads oldest first
    try:
        result, usage = ask(lines)
    except Exception as e:  # the API down or unreachable past the SDK's retries
        log.warning("lessons distill failed: %s", e)
        return _failed(conn, f"Claude couldn't be reached ({type(e).__name__}).", auto)
    usd = cost(usage)
    db.add_spend(conn, db.DISTILL_SPEND, usd)
    if result is None:
        return _failed(conn, "Claude didn't give a summary this time.", auto)
    points = [p.strip().lstrip("-• ").strip() for p in result.points]
    points = [p for p in points if p]
    if not points:
        return _failed(conn, "Claude's summary came back empty.", auto)
    # Code-side guard (a "make it shorter" call can come back longer): the summary must be a real reduction.
    if len(points) > MAX_POINTS or words(*points) >= words(*lines):
        return _failed(conn, f"Claude's summary wasn't shorter than the notes themselves ({words(*lines)} to "
                             f"{words(*points)} words), so nothing was saved.", auto)
    draft = {
        "text": "\n".join(f"- {p}" for p in points),
        "changes": result.changes.strip(),
        "at": datetime.now().replace(microsecond=0).isoformat(),
        "model": usage.get("model") or MODEL,
        "usd": round(usd, 6),
        "notes": len(lessons),
        "words": {"before": words(*lines), "after": words(*points)},
        "seen": fingerprint(lessons),
        "auto": bool(auto),
    }
    db.set_value(conn, DRAFT, _dumps(draft))
    db.set_value(conn, DISTILLED, _dumps(draft["seen"]))
    db.delete_value(conn, FAILED)
    log.info("lessons distilled: %d notes into %d points (%s, $%.4f)%s", len(lessons), len(points), draft["model"],
             usd, " automatically" if auto else "")
    return {"ok": True, "draft": draft}


def _failed(conn, error: str, auto: bool) -> dict:
    if auto:
        db.set_value(conn, FAILED, _dumps({"at": datetime.now().replace(microsecond=0).isoformat(), "error": error}))
    return {"ok": False, "error": error}


def accept(conn, text: str | None = None) -> bool:
    """Make the draft (or the owner's edited version of it) the summary Claude reads. False with no draft."""
    draft = db.get_json(conn, DRAFT)
    if not draft:
        return False
    final = _tidy(draft["text"] if text is None else text)
    meta = {k: draft.get(k) for k in ("at", "model", "usd", "notes", "seen")}
    meta["accepted_at"] = datetime.now().replace(microsecond=0).isoformat()
    meta["edited"] = final != _tidy(draft["text"])
    db.save_settings(conn, lessons_summary=final)
    db.set_value(conn, SUMMARY_META, _dumps(meta))
    db.delete_value(conn, DRAFT)
    return True


def discard(conn) -> None:
    db.delete_value(conn, DRAFT)


def _tidy(text: str) -> str:
    """The owner's textarea as prompt lines: one point per line, each starting with "- ", blank lines dropped."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-• ").strip()
        if line:
            out.append(f"- {line}")
    return "\n".join(out)


def pending_auto_draft(conn) -> dict | None:
    """The draft an automatic distill left for the owner, for the pop-up; None when there is none or it was manual."""
    draft = db.get_json(conn, DRAFT)
    return draft if draft and draft.get("auto") else None


_lock = threading.Lock()
_running = False


def maybe_auto(conn) -> bool:
    """Start an automatic distill when AUTO_EVERY new notes have been saved since the last one, there is no
    draft waiting and the last failure is old enough. Runs in a daemon thread with its own connection (the
    request returns at once); one at a time. Returns whether one was started."""
    global _running
    if db.get_json(conn, DRAFT):
        return False
    failed = db.get_json(conn, FAILED)
    if failed and _age(failed.get("at")) < RETRY_AFTER:
        return False
    if new_since(conn) < AUTO_EVERY:
        return False
    with _lock:
        if _running:
            return False
        _running = True
    if not BACKGROUND:
        _run()
        return True
    threading.Thread(target=_run, name="lessons-distill", daemon=True).start()
    return True


def _run() -> None:
    global _running
    try:
        with db.session() as conn:
            distill(conn, auto=True)
    except Exception:  # never let the thread die noisily or leave _running stuck
        log.exception("automatic lessons distill crashed")
    finally:
        with _lock:
            _running = False


def _age(at: str | None) -> float:
    if not at:
        return float("inf")
    try:
        return max(0.0, (datetime.now() - datetime.fromisoformat(at)).total_seconds())
    except ValueError:
        return float("inf")


def _dumps(value) -> str:
    return json.dumps(value)
