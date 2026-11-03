"""Species names for the correction form: search the eBird list as you type, and "Ask Claude" for a name from a clue.

Names come from the eBird taxonomy (every bird species, official English names), downloaded once to
DATA_DIR/ebird_names.json. Picking from it keeps one spelling per species, so the owner's "house finch" and Claude's
"House Finch" are the same species everywhere on the site.
"""

import re
import time
from datetime import datetime

from pydantic import BaseModel

from . import config, db

MODEL = "claude-sonnet-5-5"
TAXONOMY_URL = "https://api.ebird.org/v2/ref/taxonomy/ebird"
_names: list[str] | None = None
_failed_at = 0.0  # time.monotonic() of the last failed download


def _key(name: str) -> str:
    """Letters only, lowercased: "House-Finch", "housefinch" and "House Finch" match."""
    return re.sub(r"[^a-z]", "", name.lower())


def bird_names(fetch: bool = True) -> list[str]:
    """Every eBird species name, downloaded on first use (~1 MB). Empty if it can't be fetched; a failed download
    is retried at most hourly (every search keystroke would otherwise wait on a 30 s request).
    With fetch=False the list is only read from memory or disk, never downloaded: the watcher mustn't wait on
    the network in the middle of a visit."""
    global _names, _failed_at
    if _names is None:
        from . import species_info
        path = config.DATA_DIR / "ebird_names.json"
        names = species_info.read_json(path)  # a truncated or garbled file counts as missing
        if not isinstance(names, list):
            if not fetch or (_failed_at and time.monotonic() - _failed_at < species_info.RETRY_AFTER):
                return []
            try:
                import httpx

                r = httpx.get(TAXONOMY_URL, params={"fmt": "json", "cat": "species"}, headers=species_info.HEADERS,
                              timeout=30)
                r.raise_for_status()
                names = [t["comName"] for t in r.json()]
                species_info.write_json_atomic(path, names)
            except Exception:
                _failed_at = time.monotonic()
                return []  # offline: the yard's own names still work; try again later
        _names = names
    return _names


def canonical(name: str, known: list[str] | None = None, fetch: bool = True) -> str | None:
    """The official spelling of a name (yard names first, then eBird), or None if it isn't a known species.
    fetch=False never downloads the eBird list (bird_names)."""
    k = _key(name)
    if not k:
        return None
    for n in [*(known or []), *bird_names(fetch)]:
        if _key(n) == k:
            return n
    return None


def search(q: str, known: list[str], limit: int = 12) -> list[str]:
    """eBird names matching q, best first: whole name starts with it, then a word does, then anywhere.
    Names in `known` (already shown by the page) are left out."""
    k = _key(q)
    if len(k) < 2:
        return []
    words = [w for w in re.split(r"[^a-z]+", q.lower()) if w]
    skip = {_key(n) for n in known}
    ranked = []
    for n in bird_names():
        nk = _key(n)
        if nk in skip:
            continue
        if nk.startswith(k):
            rank = 0
        elif all(any(w2.startswith(w) for w2 in re.split(r"[^a-z]+", n.lower())) for w in words):
            rank = 1  # "finch house", "hou fin"
        elif k in nk:
            rank = 2
        else:
            continue
        ranked.append((rank, len(n), n))
    return [n for _, _, n in sorted(ranked)[:limit]]


PROMPT = """You help the owner of a backyard wildlife camera name an animal in one of its photos. They give you a clue
(what they saw, a rough or partial name, or a color or mark). Find the animal the clue points to and give its
standard English name: for a bird, the exact eBird/Clements common name ("House Finch", not "Western House Finch"
or "house finch"); for a mammal, its common species name ("Eastern Gray Squirrel"). Use the location and date to pick
between look-alikes. If the clue doesn't fit anything in the photo, say what you do see and lower the confidence.
reason: one short line for the owner on what in the photo decided it."""


class Suggestion(BaseModel):
    name: str
    visitor: str  # bird, squirrel, raccoon, possum, other
    confidence: str  # high, medium, low
    reason: str


def ask(jpeg_b64: str, hint: str, captured_at: datetime, location: str = "") -> tuple[Suggestion | None, dict]:
    """Sonnet's name for the animal the clue describes, plus token use."""
    from .identify import client

    text = (f"Clue: {hint}\nPhoto taken {captured_at:%B %-d, %Y at %-I:%M %p}."
            + (f"\nLocation: {location}" if location else ""))
    response = client().messages.parse(
        model=MODEL,
        max_tokens=1000,
        output_config={"effort": "low"},
        system=PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64}},
            {"type": "text", "text": text}]}],
        output_format=Suggestion,
    )
    usage = {"model": response.model, "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return None, usage
    return response.parsed_output, usage


def cost(usage: dict) -> float:
    price = db.USD_PER_MTOK["sonnet"]
    return (usage.get("input_tokens", 0) * price["input"] + usage.get("output_tokens", 0) * price["output"]) / 1e6
