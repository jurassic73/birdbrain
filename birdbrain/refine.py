"""Refine with Claude: Haiku rewrites the owner's description boxes into short, plain lines for the identify prompt.

The result is only a suggestion: the page puts it back into the boxes and the owner saves it (or not).
The SDK is imported lazily so the web app doesn't load it at startup.
"""

from pydantic import BaseModel

from . import db

MODEL = "claude-haiku-5-5"

PROMPT = """You tidy up the notes an owner wrote about their backyard wildlife camera. The notes are pasted into the
prompt of another model that identifies animals in each photo, so they should be short and plain but keep every fact
that helps it.

You get three boxes. Return the same three, rewritten:
- scene_notes: what the camera sees. The feeder or birdbath, its size and distance, what birds land on, the ground
  and background. Permanent things only.
- target_notes: what to look for. The animals expected here, their sizes against things in view, where they land or
  pass, what counts as a visit.
- ignore_notes: things that move or look like animals but are not, and when (night vision, after a refill, wind).

Rules:
- Keep every size, count, position and color the owner gave. Never add facts that aren't in the notes.
- Each fact goes in exactly one box. Never repeat a fact in two boxes.
- One fact per line, each line on its own line (keep line breaks). No filler ("please", "it is important that").
- The location is context only. It has its own field and is already sent; never copy it into a box.
- Drop instructions about the answer's format (e.g. "respond with NO_BIRD"); the other model's format is set elsewhere.
- If the owner's facts contradict each other, keep the more specific one and mention it in changes.
- An empty box stays empty unless facts from another box belong there.
- Your three boxes together must have fewer words than the owner's, never more. Cut words, not facts.
- If the notes are already short and in the right boxes, return them unchanged and say so in changes.

changes: one short line on what you changed, for the owner."""


class Refined(BaseModel):
    scene_notes: str
    target_notes: str
    ignore_notes: str
    changes: str


def ask(location: str, scene_notes: str, target_notes: str, ignore_notes: str) -> tuple[Refined | None, dict]:
    """Haiku's rewrite of the three boxes, plus token use."""
    from .identify import client

    text = (f"Location (context only, don't copy it): {location or '(not given)'}\n\nWhat the camera sees:\n{scene_notes or '(empty)'}\n\n"
            f"What to look for:\n{target_notes or '(empty)'}\n\nThings to ignore:\n{ignore_notes or '(empty)'}")
    response = client().messages.parse(
        model=MODEL,
        max_tokens=3000,
        output_config={"effort": "low"},
        system=PROMPT,
        messages=[{"role": "user", "content": text}],
        output_format=Refined,
    )
    usage = {"model": response.model, "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return None, usage
    return response.parsed_output, usage


def cost(usage: dict) -> float:
    price = db.USD_PER_MTOK["haiku"]
    return (usage.get("input_tokens", 0) * price["input"] + usage.get("output_tokens", 0) * price["output"]) / 1e6
