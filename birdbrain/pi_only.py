"""Pi only: what the Pi decides about a frame when Claude is not asked.

Every frame that gets past the Pi animal check and that the BioCLIP 2 species check does not name at its usual bars
goes to Claude. This module says what the Pi would do with such a frame on its own, from what the watcher already
has in hand (no new model, no extra look):

- named: the tight look names it at a lower bar (config.PI_ONLY_MIN_SCORE; a species settled here, mostly on local
  species, by day), or the first look was sure and only the crowd counter (several animals) sent it on;
- held: nothing could name it but the Pi animal check is sure something is there (config.PI_ONLY_HOLD_SCORE), or
  the check gave no verdict at all: it waits for the owner in the review queue, hidden from visitors;
- empty: filed as a false trigger.

The mode (mode(): picked on Settings > Pi checks under step 3, Claude; else PI_ONLY in .env) says what happens
with the verdict: "shadow" (the default) stores it beside Claude's answer as raw_json.calls.pi_only, scored on Stats
by db.pi_only_report, and changes nothing; "on" acts on it and Claude is never asked by the watcher; "off" does
neither.

Where the numbers come from (2026-11-17, every stored photo checked by the owner, replayed on the 183 frames that
went to Claude since 2026-11-07: 18 animals, 165 empty): the tight look at 0.75 named 6 of the animals and nothing
wrongly, and no empty frame (on those its top guess never passed 0.83, always a species not settled here); a Pi
animal score of 0.9 or more held 4 more animals and 1 empty frame. Left for Claude alone: gray squirrels on the
ground that no finder pointed at. No night visitor has come yet, so the night side of this is untested.
"""

from __future__ import annotations

from . import classifier, config, db

NAMED, HELD, EMPTY = "named", "held", "empty"
MODES = ("off", "shadow", "on")
MODE_KEY = "pi_only_mode"  # the settings row the Settings page writes


def mode(conn) -> str:
    """The mode picked on Settings > Pi checks, else PI_ONLY from .env (like presence.mode)."""
    value = db.get_value(conn, MODE_KEY)
    return value if value in MODES else config.PI_ONLY


def verdict(check: dict | None, tight: dict | None, crowd: str | None, settled: dict[str, str], night: bool) -> dict:
    """What the Pi alone would do with a frame the gate sent on to Claude. `check` is the Pi animal check's verdict
    (None when it is off or failed), `tight` the tight look's guess when one was made, `crowd` the species the first
    look was sure of when several animals in view stopped it from naming the visit, `settled` db.settled_species.
    {"verdict": named/held/empty, "score": the animal check's score or None, and for named "species", "visitor" and
    "by" ("tight" or "bowl"), for held "why" ("score" or "unchecked")}."""
    score = check.get("score") if check else None
    out = {"verdict": EMPTY, "score": score}
    if crowd and settled.get(crowd) and not night:
        # The first look cleared its own bar; only the count is in doubt, and a name without a count beats none.
        return {**out, "verdict": NAMED, "species": crowd, "visitor": settled[crowd], "by": "bowl"}
    pick = classifier.gate_pick(tight, settled, night, view="tight", bar=config.PI_ONLY_MIN_SCORE) if tight else None
    if pick:
        return {**out, "verdict": NAMED, "species": pick[0], "visitor": pick[1], "by": "tight"}
    if score is None:
        return {**out, "verdict": HELD, "why": "unchecked"}  # no local verdict at all: never lose it quietly
    if score >= config.PI_ONLY_HOLD_SCORE:
        return {**out, "verdict": HELD, "why": "score"}
    return out
