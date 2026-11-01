"""Identify the visitor in a frame with Claude. CLI: python -m birdbrain.identify photo.jpg"""

import base64
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import anthropic
from PIL import Image
from pydantic import BaseModel

from . import config

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
MAX_EDGE = 1568  # the API downsizes anything larger, so don't upload the extra pixels

SYSTEM = """You identify animals in photos from a backyard wildlife camera, usually pointed at a birdbath or feeder.

Fill in every field:
- visitor: "bird", "squirrel", "raccoon", "possum", "other" (cats, people, deer, rats, anything else), or "none" when no animal is visible. Moving leaves, shadows, water and insects are "none".
- species: for birds, the standard eBird/Cornell common name, e.g. "Dark-eyed Junco". For mammals, a specific common name if clear (e.g. "Eastern Gray Squirrel", "Douglas Squirrel"), otherwise null. Null for "other" and "none".
- count: how many of that visitor are visible; 0 for "none".
- confidence: "high" only when diagnostic field marks are clearly visible; "medium" when the ID is likely but marks are partly hidden, blurry or distant; "low" when it is a guess.
- notes: one short line: subspecies or race, sex, age, plumage, behavior, or what makes the ID uncertain.
- circle: a circle around the animal you identified, in the first photo: x and y are its center as fractions of the photo's width and height (0 to 1, from the top left), r its radius as a fraction of the width. Make it snug but include the whole animal. Null for "none".
- also: when more than one kind of animal is visible, the fields above describe the most prominent one and every other kind gets one entry here, with its own visitor, species, count, confidence and a circle around one of them. Two of the same species are one entry with count 2. Empty when only one kind is visible, and always empty for "none".

Night frames are grayscale infrared; color cannot be used then. Weigh what is common in the region for the date given, but report an unusual species if the field marks clearly support it. Fallen leaves are not animals."""


class Circle(BaseModel):
    x: float
    y: float
    r: float


class Also(BaseModel):
    """Another kind of animal in the same photo, besides the main one."""
    visitor: Literal["bird", "squirrel", "raccoon", "possum", "other"]
    species: str | None
    count: int
    confidence: Literal["high", "medium", "low"]
    circle: Circle | None = None


class Sighting(BaseModel):
    visitor: Literal["bird", "squirrel", "raccoon", "possum", "other", "none"]
    species: str | None
    count: int
    confidence: Literal["high", "medium", "low"]
    notes: str
    circle: Circle | None = None
    also: list[Also] = []


def frame_circle(s: Sighting | Also, crop=None) -> dict | None:
    """Claude's circle moved from the cropped photo it saw onto the full frame, as the visit page's marks use."""
    c = s.circle
    if s.visitor == "none" or c is None or not (0 <= c.x <= 1 and 0 <= c.y <= 1 and 0 < c.r <= 1):
        return None
    x, y, w, h = crop or (0, 0, 1, 1)
    return {"x": round(x + c.x * w, 4), "y": round(y + c.y * h, 4), "r": round(max(c.r * w, 0.01), 4), "claude": True}


def also_rows(s: Sighting, crop=None) -> list[dict]:
    """The other kinds of animal in the photo, as stored in visits.also, with circles moved onto the full frame.
    Entries repeating the main animal or each other are dropped, and a "none" answer has none."""
    if s.visitor == "none":
        return []
    rows, seen = [], {(s.visitor, (s.species or "").lower())}
    for a in s.also:
        key = (a.visitor, (a.species or "").lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"visitor": a.visitor, "species": a.species or None, "count": max(a.count, 1),
                     "confidence": a.confidence, "circle": frame_circle(a, crop)})
    return rows


@dataclass
class Result:
    sighting: Sighting
    model: str
    escalated: bool
    needs_review: bool
    raw: dict


_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(timeout=60.0)
    return _client


def prepare(jpeg: bytes, crop: tuple[float, float, float, float] | None = None) -> str:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if crop:
        x, y, w, h = crop
        W, H = img.size
        img = img.crop((int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)))
    img.thumbnail((MAX_EDGE, MAX_EDGE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode()


def system_prompt(lessons: list[str] | None = None, site: dict | None = None) -> str:
    """SYSTEM plus Claude's Notes (location, the owner's description, what to look for, what to ignore)
    and recent corrections."""
    site = site or {}
    parts = [SYSTEM]
    if site.get("location"):
        parts.append(f"The camera is in {site['location']}. Use that region's birds and mammals for the date given.")
    if site.get("scene_notes"):
        parts.append("What this camera sees, in the owner's words:\n" + site["scene_notes"])
    if site.get("target_notes"):
        parts.append("What to look for here, in the owner's words:\n" + site["target_notes"])
    if site.get("ignore_notes"):
        parts.append("Things here that move or look like animals but are not:\n" + site["ignore_notes"])
    if lessons:
        parts.append(
            "Corrections the owner made to your recent answers at this camera, newest first. "
            "Learn from them, especially about what is not an animal here:\n" + "\n".join(lessons)
        )
    return "\n\n".join(parts)


def user_content(images_b64: list[str], captured_at: datetime) -> list[dict]:
    when = f"{captured_at:%B %-d, %Y at %-I:%M %p} local time"
    if len(images_b64) == 1:
        text = f"Photo taken {when}."
    else:
        text = (f"These {len(images_b64)} photos are from the same visit, taken a fraction of a second apart "
                f"on {when}. Combine what each one shows (face, breast, wings, tail) into one identification.")
    blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b}} for b in images_b64]
    return [*blocks, {"type": "text", "text": text}]


def ask(
    model: str,
    images_b64: list[str],
    captured_at: datetime,
    lessons: list[str] | None = None,
    site: dict | None = None,
) -> tuple[Sighting | None, dict]:
    response = client().beta.messages.parse(
        model=model,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low" if model == HAIKU else "medium"},
        system=system_prompt(lessons, site),
        messages=[{"role": "user", "content": user_content(images_b64, captured_at)}],
        output_format=Sighting,
    )
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return None, {"model": response.model, "stop_reason": response.stop_reason, **usage}
    return response.parsed_output, {"model": response.model, **usage}


def needs_second_look(s: Sighting | None, confirmed: set[str] = frozenset()) -> bool:
    """Escalate anything short of a confident answer about an actual animal, except a medium answer
    naming a species already confirmed here several times (db.confirmed_species): on 2026-10-27 Sonnet
    agreed with every such Haiku answer, while the wrong Haiku guesses were all new names.
    A low-confidence guess at a second animal in the photo gets the second look too."""
    if s is None or s.visitor == "none":
        return s is None
    if any(a.confidence == "low" for a in s.also):
        return True
    if s.confidence == "high":
        return False
    return not (s.confidence == "medium" and s.species in confirmed)


def identify(
    jpeg: bytes,
    captured_at: datetime,
    crop=None,
    lessons: list[str] | None = None,
    extra_frames: list[bytes] = (),
    site: dict | None = None,
    confirmed: set[str] = frozenset(),
) -> Result:
    """Haiku looks at the best frame; if unsure, Sonnet gets that frame plus the next-best burst frames."""
    image_b64 = prepare(jpeg, crop)
    sighting, meta = ask(HAIKU, [image_b64], captured_at, lessons, site)
    raw = {"haiku": {**meta, "answer": sighting.model_dump() if sighting else None}, "lessons": len(lessons or [])}
    model, escalated = HAIKU, False
    if not needs_second_look(sighting, confirmed) and needs_second_look(sighting):
        raw["skipped_sonnet"] = "confirmed species"
    elif needs_second_look(sighting, confirmed):
        images = [image_b64, *(prepare(f, crop) for f in extra_frames)]
        second, meta2 = ask(SONNET, images, captured_at, lessons, site)
        raw["sonnet"] = {**meta2, "frames": len(images), "answer": second.model_dump() if second else None}
        if second is not None:
            sighting, model, escalated = second, SONNET, True
    if sighting is None:
        sighting = Sighting(visitor="other", species=None, count=0, confidence="low", notes="Model declined to answer")
    also = also_rows(sighting, crop)
    needs_review = sighting.visitor != "none" and (
        sighting.confidence == "low" or any(a["confidence"] == "low" for a in also))
    if circle := frame_circle(sighting, crop):
        raw["circle"] = circle
    if also:
        raw["also"] = also
    return Result(sighting, model, escalated, needs_review, raw)


LOCAL_MODEL = "yolox-tiny"


def settled_locally(det: dict) -> Result:
    """Gate mode: the local detector found nothing, so Claude isn't asked."""
    notes = "Local detector found no animal; Claude was not asked."
    s = Sighting(visitor="none", species=None, count=0, confidence="medium", notes=notes)
    return Result(s, LOCAL_MODEL, escalated=False, needs_review=False, raw={"detector": det, "skipped_claude": True})


def settled_by_presence(check: dict, det: dict | None) -> Result:
    """Gate mode of the local animal check (birdbrain.presence): it judged the frame empty, so Claude isn't
    asked. Stored as a false trigger for the owner's daily review."""
    notes = (f"Local animal check found nothing (score {check['score']:.3f}, threshold {check['threshold']:.3f}); "
             "Claude was not asked.")
    s = Sighting(visitor="none", species=None, count=0, confidence="medium", notes=notes)
    raw = {"presence": check, "skipped_claude": True, **({"detector": det} if det is not None else {})}
    return Result(s, check["model"], escalated=False, needs_review=False, raw=raw)


def settled_by_classifier(guess: dict, pick: tuple[str, str], check: dict | None) -> Result:
    """Gate mode of the local species classifier: a confident daytime guess of a species already settled here
    names the visit, and Claude isn't asked. It can't count or see a second kind of animal."""
    species, visitor = pick
    score = guess["top"][0]["score"]
    notes = (f"Named on the Pi by the local classifier ({score:.0%} sure); Claude was not asked, "
             "so the count and any other animals weren't checked.")
    s = Sighting(visitor=visitor, species=species, count=1, confidence="high", notes=notes)
    raw = {"classifier": guess, "skipped_claude": True, "settled_by": "classifier",
           **({"presence": check} if check else {})}
    return Result(s, guess["model"], escalated=False, needs_review=False, raw=raw)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m birdbrain.identify photo.jpg [more.jpg ...]")
    from . import db  # the CLI uses the same Settings and lessons as the live pipeline

    with db.session() as conn:
        db.init_db(conn)
        site, lessons = db.prompt_site(conn), db.recent_lessons(conn)
    for path in sys.argv[1:]:
        result = identify(open(path, "rb").read(), datetime.now(), config.CROP_ROI, lessons, site=site)
        print(path)
        print(json.dumps({"answer": result.sighting.model_dump(), "model": result.model,
                          "escalated": result.escalated, "needs_review": result.needs_review,
                          "calls": result.raw}, indent=2))


if __name__ == "__main__":
    main()
