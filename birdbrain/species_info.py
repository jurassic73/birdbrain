"""Species info cards: reference photo, description, local status and seasonality, range map, behavior notes, links.

Built once per species from free sources (iNaturalist, Wikipedia, the eBird taxonomy) plus one small Claude Haiku
call, cached in the species_info table and refreshed monthly. Any source can fail; the card shows what it got.
"""

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from urllib.parse import quote, unquote

import httpx
from pydantic import BaseModel

from . import config, db

log = logging.getLogger("birdbrain.species_info")

# Wikipedia answers 403 to httpx's default Accept-Encoding (brotli, zstd); plain gzip is accepted.
HEADERS = {"User-Agent": "Birdbrain/0.1 (personal backyard bird camera)", "Accept-Encoding": "gzip"}
REFRESH_AFTER = timedelta(days=30)
NO_CARD = {"None", "Other"}
# Generic mammal labels Claude uses when it can't name the species; these are the likely ones in North America.
ALIASES = {"Possum": "Virginia Opossum", "Raccoon": "Common Raccoon"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
EXPECTED_SHARE = 0.10  # a month is "expected" with at least this share of the peak month's reports
MIN_LOCAL_REPORTS = 20
LICENSES = {
    "cc0": "CC0", "cc-by": "CC BY", "cc-by-sa": "CC BY-SA", "cc-by-nc": "CC BY-NC",
    "cc-by-nc-sa": "CC BY-NC-SA", "cc-by-nd": "CC BY-ND", "cc-by-nc-nd": "CC BY-NC-ND",
}

_building: set[str] = set()
RETRY_AFTER = 3600  # seconds before trying a failed download or rebuild again
MERGED_FIELDS = ("description", "notes", "facts", "photo", "sounds", "map_center")  # a rebuild lacking one keeps the old
_failed_at: dict[str, float] = {}  # species name (or "ebird_codes") -> time.monotonic() of its last failed fetch
_lock = threading.Lock()


def has_card(name: str | None) -> bool:
    return bool(name) and name not in NO_CARD


def _get(client: httpx.Client, url: str, **params) -> dict:
    r = client.get(url, params=params or None, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


PACE = 2  # seconds between cards in the facts backfill, to stay under Wikipedia's rate limit
PATIENT_WAITS = (10, 30)  # seconds before the 1st and 2nd retry of a 429/503 when there is no Retry-After
RETRY_AFTER_MAX = 120
_sleep = time.sleep  # tests replace this so nothing really waits


def _get_patient(client: httpx.Client, url: str, **params) -> dict:
    """Like _get, but a 429 or 503 is waited out (Retry-After if given, else 10 s then 30 s) and retried twice."""
    for attempt in range(len(PATIENT_WAITS) + 1):
        r = client.get(url, params=params or None, headers=HEADERS, timeout=15)
        if r.status_code in (429, 503) and attempt < len(PATIENT_WAITS):
            wait = PATIENT_WAITS[attempt]
            retry = (r.headers.get("Retry-After") or "").strip()
            if retry.isdigit():
                wait = min(int(retry), RETRY_AFTER_MAX)
            log.warning("%s from %s, waiting %s s", r.status_code, url.split("?")[0], wait)
            _sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


def find_taxon(client: httpx.Client, query: str) -> dict | None:
    """The iNaturalist taxon for a common name: an exact common-name match at species rank if there is one."""
    results = _get(client, "https://api.inaturalist.org/v1/taxa", q=query, is_active="true", per_page=10)["results"]
    if not results:
        return None
    q = query.lower()
    exact = [t for t in results if (t.get("preferred_common_name") or "").lower() == q]
    for pool in (exact, results):
        for t in pool:
            if t.get("rank") == "species":
                return t
    return (exact or results)[0]


def open_photo(client: httpx.Client, taxon: dict) -> dict:
    """The taxon's first openly licensed photo; the default one is sometimes all rights reserved."""
    default = taxon.get("default_photo") or {}
    if default.get("license_code"):
        return default
    try:
        full = _get(client, f"https://api.inaturalist.org/v1/taxa/{taxon['id']}")["results"][0]
    except Exception:
        log.exception("iNaturalist photo lookup failed")
        return {}
    for tp in full.get("taxon_photos") or []:
        if (tp.get("photo") or {}).get("license_code"):
            return tp["photo"]
    return {}


def local_status(counts: list[int]) -> tuple[str, list[int]]:
    """(status line, expected months 1-12) from a year of local monthly report counts."""
    total, peak = sum(counts), max(counts or [0])
    if total < MIN_LOCAL_REPORTS:
        return "Rarely reported here", []
    expected = [m for m, n in enumerate(counts, 1) if n >= EXPECTED_SHARE * peak]
    summer_share = sum(counts[3:9]) / total  # Apr to Sep
    if len(expected) >= 10:
        return "Year-round resident", expected
    if summer_share >= 0.8:
        return "Summer visitor", expected
    if summer_share <= 0.2:
        return "Winter visitor", expected
    if len(expected) <= 7:
        return "Passes through in spring and fall", expected
    return "Year-round resident", expected


def links(common: str, is_bird: bool, ebird_code: str | None, inat_id: int | None, wiki_url: str | None) -> list[dict]:
    out = []
    if is_bird:
        out.append({"label": "All About Birds", "url": f"https://www.allaboutbirds.org/guide/{quote(common.replace(' ', '_'))}"})
        slug = re.sub(r"[^a-z0-9]+", "-", common.lower().replace("'", "")).strip("-")
        out.append({"label": "Audubon", "url": f"https://www.audubon.org/field-guide/bird/{slug}"})
        if ebird_code:
            out.append({"label": "eBird", "url": f"https://ebird.org/species/{ebird_code}"})
    if wiki_url:
        out.append({"label": "Wikipedia", "url": wiki_url})
    if inat_id:
        out.append({"label": "iNaturalist", "url": f"https://www.inaturalist.org/taxa/{inat_id}"})
    return out


def write_json_atomic(path, obj) -> None:
    """Write JSON so a crash or full disk never leaves a truncated file: write a .part file, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        part.write_text(json.dumps(obj))
        os.replace(part, path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def read_json(path):
    """The parsed file, or None if it is missing or unreadable (a truncated download counts as missing)."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def ebird_code(client: httpx.Client, common: str) -> str | None:
    """eBird species code by common name, from the eBird taxonomy (downloaded once, ~3.5 MB, no key needed).
    A failed download is retried at most hourly."""
    path = config.DATA_DIR / "ebird_codes.json"
    codes = read_json(path)
    if not isinstance(codes, dict):
        last = _failed_at.get("ebird_codes")
        if last is not None and time.monotonic() - last < RETRY_AFTER:
            return None
        try:
            taxa = _get(client, "https://api.ebird.org/v2/ref/taxonomy/ebird", fmt="json", cat="species")
            codes = {t["comName"].lower(): t["speciesCode"] for t in taxa}
            write_json_atomic(path, codes)
        except Exception:
            _failed_at["ebird_codes"] = time.monotonic()
            raise
    return codes.get(common.lower())


QUALITY = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
# Distance bands beat recording quality, so birds sound like the local ones (dialects differ by region).
LOCAL_KM = 150
NEARBY_KM = 500


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def _https(url: str) -> str:
    return "https:" + url if url.startswith("//") else url


def license_name(url: str) -> str:
    """'//creativecommons.org/licenses/by-nc-sa/4.0/' -> 'CC BY-NC-SA 4.0'."""
    m = re.search(r"licenses/([a-z-]+)/([\d.]+)", url or "")
    return f"CC {m.group(1).upper()} {m.group(2)}" if m else (url or "")


def pick_sounds(recordings: list[dict], center: list[float] | None, n: int = 3) -> list[dict]:
    """A song, a call, and one more, preferring recordings near home, then better quality."""
    ranked = []
    for r in recordings:
        if not r.get("file"):
            continue
        try:
            where = (float(r["lat"]), float(r.get("lon") or r.get("lng")))
            km = _km(tuple(center), where) if center else None
        except (KeyError, TypeError, ValueError):
            km = None
        band = 2 if km is None or km > NEARBY_KM else 1 if km > LOCAL_KM else 0
        ranked.append(((band, QUALITY.get(r.get("q"), 5), km if km is not None else 1e9), km, r))
    ranked.sort(key=lambda t: t[0])
    types = lambda t: (t[2].get("type") or "").lower()  # noqa: E731
    chosen: list[tuple] = []
    for kind, other in (("song", "call"), ("call", "song")):
        # a pure song (or pure call) first, then a mixed recording
        hit = next((t for t in ranked if kind in types(t) and other not in types(t) and t not in chosen), None)
        hit = hit or next((t for t in ranked if kind in types(t) and t not in chosen), None)
        if hit:
            chosen.append(hit)
    chosen += [t for t in ranked if t not in chosen][: n - len(chosen)]
    out = []
    for _, km, r in chosen[:n]:
        kind = (r.get("type") or "").lower()
        out.append({
            "id": r.get("id"),
            "label": ("Song and call" if "song" in kind and "call" in kind else "Song" if "song" in kind
                      else "Call" if "call" in kind else (r.get("type") or "Recording").capitalize()),
            "file": _https(r["file"]),
            "page": f"https://xeno-canto.org/{r.get('id')}",
            "recordist": r.get("rec") or "unknown",
            "license": license_name(r.get("lic") or ""),
            "where": ", ".join(p for p in (r.get("loc"), r.get("cnt")) if p),
            "km": round(km) if km is not None else None,
            "length": r.get("length"),
        })
    return out


def fetch_sounds(client: httpx.Client, sci_name: str | None, center: list[float] | None) -> list[dict]:
    parts = (sci_name or "").split()
    if len(parts) < 2:
        return []
    base = f"gen:{parts[0]} sp:{parts[1]} grp:birds"

    def search(query: str) -> list[dict]:
        try:
            data = _get(client, "https://xeno-canto.org/api/3/recordings", query=query, key=config.XC_API_KEY, per_page=200)
        except httpx.HTTPError as e:
            # The API key rides in the URL, and httpx puts the URL in its error text: say only what went wrong.
            status = getattr(getattr(e, "response", None), "status_code", None)
            raise RuntimeError(f"xeno-canto answered {status or type(e).__name__} for {sci_name}") from None
        return data.get("recordings") or []

    # Common birds have hundreds of recordings and the API returns only 200, so ask for ones near home first.
    recordings = []
    if center:
        lat, lon = center
        dlat = NEARBY_KM / 111
        dlon = dlat / max(math.cos(math.radians(lat)), 0.1)
        recordings = search(f"{base} box:{lat - dlat:.2f},{lon - dlon:.2f},{lat + dlat:.2f},{lon + dlon:.2f}")
    if len(recordings) < 3:
        seen = {r.get("id") for r in recordings}
        recordings += [r for r in search(base) if r.get("id") not in seen]
    return pick_sounds(recordings, center)


class Fact(BaseModel):
    lead: str
    text: str


class CardText(BaseModel):
    notes: list[str]
    facts: list[Fact]


NOTES_PROMPT = """You write short text for a backyard wildlife site, using only the article text you are given.
Never add a fact that is not in the text, and never mention "the article" or "the text".

notes: 2 to 4 notes, each one sentence of at most 25 words, in plain language:
diet; how it behaves at a feeder or birdbath (or in a backyard); how it gets along with other animals (aggressive, territorial, social).
Skip any topic the article does not cover.

facts: 2 to 5 hook facts for the top of the species page, to catch a curious visitor's interest. Put the article's most
surprising and memorable facts here (the notes are the practical behavior). Each has a text and a lead. The text is one complete sentence of 6 to 14 words with its own subject (the animal's name, "it",
or "they"), never a fragment such as "Longest recorded lifespan for females". It makes exactly one claim, taken from one
place in the article: never merge two separate claims, or a claim and a detail from a different sentence, into one fact.
Make it punchy and memorable: a size or weight comparison, a lifespan record, a speed, an odd habit, where its name comes from, a striking number.
Keep every number exactly as the article states it, and say exactly who it applies to (do not turn a claim about females
or the whole species into one about males, or the reverse). Prefer the surprising over the descriptive: skip plain
taxonomy ("a small sparrow") and ordinary size ranges unless there is a vivid comparison. A weight comparison must name a real US coin, never just "a coin": penny 2.5 g, dime 2.27 g, nickel 5 g,
quarter 5.67 g, dollar coin 8.1 g, half dollar 11.34 g. Pick the coin closest to the stated weight (for a range, the coin
inside it), and say "about" or "roughly".
They must not repeat or rephrase any
of the notes. Write fewer, or none, rather than pad with dull or generic facts. No emoji.
Every fact must have a lead. The lead is the fact's headline for a tile: 1 or 2 words, or a number with its unit, copied exactly
from the fact's own text (for example "Blizzard", "75%", "40 million", "1741", "12 years"). If a fact has no good lead in its own
sentence, choose a different fact instead. Never a word that is not in the text."""

VAGUE_COIN = re.compile(r"\b(?:a|an|the|one|two|three|four|five)\s+(?:small\s+|single\s+)?coins?\b", re.I)


def vague_coin(facts) -> bool:
    """True if a fact compares to "a coin" without naming which one (penny, nickel, dime, quarter, dollar, half dollar)."""
    for fact in facts or []:
        text = _fact_parts(fact)[1]
        if VAGUE_COIN.search(text) and not re.search(r"\b(?:penny|pennies|nickel|dime|quarter|dollar|cent)s?\b", text, re.I):
            return True
    return False


MAX_NOTES = 4
MAX_FACTS = 5
MAX_FACT_WORDS = 20  # the prompt asks for 6 to 14; this only catches a runaway sentence
FACT_DUP_SHARE = 0.6  # a fact sharing this much of its words with a note is a repeat
_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "it", "its", "as", "at", "by",
         "can", "has", "have", "from", "that", "this", "their", "they", "up", "than", "often", "about"}


def _words(text: str) -> set[str]:
    words = (w.rstrip("'") for w in re.findall(r"[a-z0-9']+", text.lower()))
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOP}  # plural-blind


LEAD_MAX_WORDS = 3
LEAD_MAX_CHARS = 18
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_UNITS = {"year", "years", "inch", "inches", "ounce", "ounces", "gram", "grams", "foot", "feet", "mph", "mile", "miles",
          "day", "days", "month", "months", "egg", "eggs", "pound", "pounds", "percent", "million", "billion", "thousand",
          "species", "kilometers", "km", "centimeters", "cm", "weeks", "week", "times"}


def _numbers(text: str) -> list[str]:
    return [n.replace(",", "").rstrip(".") for n in _NUM.findall(text)]


def number_lead(text: str) -> str:
    """A lead built in code: the first number in the fact with its % or unit word, else empty (the tile shows no lead)."""
    m = _NUM.search(text)
    if not m:
        return ""
    lead = m.group(0).rstrip(".")
    rest = text[m.end():]
    if rest.startswith("%"):
        lead += "%"
    else:
        nxt = re.match(r"[\s-]+([A-Za-z]+)", rest)
        if nxt and nxt.group(1).lower() in _UNITS:
            lead += " " + nxt.group(1)
    return lead if len(lead) <= LEAD_MAX_CHARS else m.group(0).rstrip(".")[:LEAD_MAX_CHARS]


def clean_lead(lead: str | None, text: str) -> str:
    """The model's tile lead if it is short and really comes from the fact (a number matches with or without its %/unit),
    else a number found in the fact, else nothing."""
    lead = " ".join((lead or "").split())
    if lead and len(lead.split()) <= LEAD_MAX_WORDS and len(lead) <= LEAD_MAX_CHARS:
        nums = _numbers(lead)
        if nums:
            have = _numbers(text)
            ok = all(n in have for n in nums)
        else:
            ok = lead.lower() in " ".join(text.lower().split())
        if ok:
            return lead
    return number_lead(text)


def _fact_parts(fact) -> tuple[str, str]:
    """(lead, text) from a Fact, a dict, or an old plain string."""
    if isinstance(fact, str):
        return "", fact
    if isinstance(fact, dict):
        return fact.get("lead") or "", fact.get("text") or ""
    return getattr(fact, "lead", "") or "", getattr(fact, "text", "") or ""


def fact_item(fact) -> dict:
    """A cached fact as {lead, text}; cards saved with plain-string facts get no lead."""
    lead, text = _fact_parts(fact)
    return {"lead": lead, "text": text}


def clean_facts(facts: list, notes: list[str]) -> list[dict]:
    """The model's hook facts as {lead, text}, checked in code: no blanks, none over MAX_FACT_WORDS words, none repeating a
    note or an earlier fact (by share of shared words); each lead must come from its text, and a fact still without one is
    dropped unless fewer than 2 would be left; at most MAX_FACTS."""
    kept: list[dict] = []
    seen = [_words(n) for n in notes]
    for fact in facts or []:
        lead, text = _fact_parts(fact)
        text = " ".join(text.split())
        if not text or len(text.split()) > MAX_FACT_WORDS:
            continue
        mine = _words(text)
        if not mine or any(len(mine & other) / len(mine) >= FACT_DUP_SHARE for other in seen):
            continue
        kept.append({"lead": clean_lead(lead, text), "text": text})
        seen.append(mine)
    led = [f for f in kept if f["lead"]]
    if len(led) >= 2:  # a fact with no lead is dropped, unless that would leave too few to show
        kept = led
    return kept[:MAX_FACTS]


def clean_notes(notes: list[str]) -> list[str]:
    """Behavior notes: no blanks, none saying (mostly) what an earlier note already said, at most MAX_NOTES."""
    kept: list[str] = []
    seen: list[set[str]] = []
    for note in notes or []:
        note = " ".join((note or "").split())
        mine = _words(note)
        if not note or not mine:
            continue
        if any(len(mine & other) / min(len(mine), len(other)) >= FACT_DUP_SHARE for other in seen):
            continue
        kept.append(note)
        seen.append(mine)
        if len(kept) == MAX_NOTES:
            break
    return kept


def card_text(name: str, article: str) -> dict:
    """One Haiku call on the Wikipedia article: the behavior notes and the header facts, both guarded in code."""
    from .identify import HAIKU, client  # lazy: keeps the web app from loading the SDK at startup

    response = client().beta.messages.parse(
        model=HAIKU,
        max_tokens=1500,
        output_config={"effort": "low"},
        system=NOTES_PROMPT,
        messages=[{"role": "user", "content": f"Animal: {name}\n\nArticle:\n{article[:12000]}"}],
        output_format=CardText,
    )
    parsed = response.parsed_output
    if not parsed:
        return {"notes": [], "facts": []}
    notes = clean_notes(parsed.notes)
    return {"notes": notes, "facts": clean_facts(parsed.facts, notes)}


def wikipedia_article(http: httpx.Client, title: str, patient: bool = False) -> tuple[dict | None, str]:
    """The Wikipedia summary (None if the page isn't a standard article) and its plain text. patient waits out rate limits."""
    get = _get_patient if patient else _get
    summary = get(http, f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}")
    if summary.get("type") != "standard":
        return None, ""
    ext = get(http, "https://en.wikipedia.org/w/api.php", action="query", prop="extracts",
              explaintext=1, redirects=1, format="json", titles=summary["title"])
    return summary, next(iter(ext["query"]["pages"].values())).get("extract", "")


def build(name: str) -> dict:
    """Gather everything for one species. Each source is optional."""
    query = ALIASES.get(name, name)
    data: dict = {"name": name, "query": query}
    with httpx.Client(follow_redirects=True) as http:
        taxon = None
        try:
            taxon = find_taxon(http, query)
        except Exception:
            log.exception("iNaturalist taxon lookup failed for %s", query)
        is_bird = bool(taxon and taxon.get("iconic_taxon_name") == "Aves")
        wiki_url = None
        if taxon:
            photo = open_photo(http, taxon)
            data.update(
                common=taxon.get("preferred_common_name") or query,
                sci_name=taxon.get("name"),
                inat_id=taxon["id"],
                is_bird=is_bird,
            )
            if photo.get("medium_url"):
                data["photo"] = {
                    "url": photo["medium_url"],
                    "credit": re.sub(r",? ?uploaded by .*$", "", photo.get("attribution") or ""),
                    "license": LICENSES.get(photo.get("license_code") or "", photo.get("license_code") or ""),
                }
            wiki_url = (taxon.get("wikipedia_url") or "").replace("http://", "https://") or None
        common = data.get("common", query)
        title = wiki_url.rsplit("/", 1)[-1] if wiki_url else common.replace(" ", "_")
        article = ""
        try:
            summary, article = wikipedia_article(http, title)
            if summary:
                data["description"] = summary.get("extract")
                wiki_url = summary["content_urls"]["desktop"]["page"]
        except Exception:
            log.exception("Wikipedia lookup failed for %s", title)
        data["wiki_url"] = wiki_url
        if taxon and config.INAT_PLACE_ID:
            try:
                hist = _get(http, "https://api.inaturalist.org/v1/observations/histogram", taxon_id=taxon["id"],
                            place_id=config.INAT_PLACE_ID, interval="month_of_year", verifiable="true")
                months = hist["results"]["month_of_year"]
                counts = [int(months.get(str(m), 0)) for m in range(1, 13)]
                status, expected = local_status(counts)
                data.update(local_counts=counts, status=status, expected_months=expected)
                place = _get(http, f"https://api.inaturalist.org/v1/places/{config.INAT_PLACE_ID}")["results"][0]
                data["place_name"] = place.get("display_name")
                lat, lng = (float(v) for v in place["location"].split(","))
                data["map_center"] = [lat, lng]
            except Exception:
                log.exception("iNaturalist seasonality failed for %s", common)
        code = None
        if is_bird:
            try:
                code = ebird_code(http, common)
            except Exception:
                log.exception("eBird taxonomy lookup failed")
            if config.XC_API_KEY:
                try:
                    data["sounds"] = fetch_sounds(http, data.get("sci_name"), data.get("map_center"))
                except Exception:
                    log.exception("xeno-canto lookup failed for %s", common)
    if article:
        try:
            data.update(card_text(common, article))
        except Exception:
            log.exception("behavior notes failed for %s", common)
    data["links"] = links(common, is_bird, code, data.get("inat_id"), wiki_url)
    return data


def season_badge(info: dict | None, month: int) -> str | None:
    """'Early arrival', 'Late departure', 'Unusual this time of year' or 'Rarely reported here' for a visit in
    `month` (1-12), from the card's local data; None when the visit is in season or there's no local data."""
    if not info or not info.get("local_counts"):
        return None
    expected = set(info.get("expected_months") or [])
    if not expected:
        return "Rarely reported here"
    if month in expected:
        return None
    nxt, prev = month % 12 + 1, (month - 2) % 12 + 1
    if nxt in expected and prev not in expected:
        return "Early arrival"
    if prev in expected and nxt not in expected:
        return "Late departure"
    return "Unusual this time of year"


# (time.monotonic() of the last read, cards); None = never read. Not 0.0: monotonic() counts from boot on Linux,
# so for the first minute after a reboot the empty cache looked fresh and every page lost its season badges.
_season_cache: tuple[float | None, dict] = (None, {})


def forget_cached_cards() -> None:
    global _season_cache
    _season_cache = (None, {})


def cached_cards() -> dict[str, dict]:
    """Every cached card by name, re-read at most once a minute (pages call this for each visit card)."""
    global _season_cache
    read_at, cards = _season_cache
    if read_at is None or time.monotonic() - read_at > 60:
        with db.session() as conn:
            cards = {r["name"]: json.loads(r["data"]) for r in conn.execute("SELECT name, data FROM species_info")}
        _season_cache = (time.monotonic(), cards)
    return cards


def visit_badge(label: str | None, captured_at: str) -> str | None:
    """Season badge for a stored visit; uses only cards already cached, never the network."""
    if not has_card(label):
        return None
    return season_badge(cached_cards().get(label), int(captured_at[5:7]))


SHORT_STATUS = {"Year-round resident": "Year-round", "Summer visitor": "Summer", "Winter visitor": "Winter",
                "Passes through in spring and fall": "Migrant", "Rarely reported here": "Rare here"}


def cached_facts(conn: sqlite3.Connection, label: str | None, limit: int = 3) -> list[str]:
    """The header facts of a species' cached card, for a small box on another page. Cache only: it never builds a card,
    fetches, or calls Haiku, so a visitor's page view can't cost anything. Empty for None/Other, no card, or no facts.
    Facts are plain sentences: the {lead, text} dicts of newer cards give their text, old plain strings pass through."""
    if not has_card(label):
        return []
    card = get(conn, label, fetch=False) or {}
    texts = (" ".join(str(fact_item(f)["text"]).split()) for f in card.get("facts") or [])
    return [t for t in texts if t][:limit]


def short_status(label: str | None) -> tuple[str, str] | None:
    """(short, full) local status for a cached card, for tight lists; never the network."""
    if not has_card(label):
        return None
    full = (cached_cards().get(label) or {}).get("status")
    return (SHORT_STATUS.get(full, full), full) if full else None


def _needs_sounds(data: dict) -> bool:
    """Cards built before the xeno-canto key was set get their sounds added on next view, no full rebuild.
    A card a viewer flagged (a recording that wouldn't load) gets its sounds fetched again the same way."""
    if not config.XC_API_KEY or not data.get("is_bird"):
        return False
    return "sounds" not in data or bool(data.get("sounds_broken"))


def flag_broken_sound(conn: sqlite3.Connection, name: str, url: str) -> bool:
    """Note that a saved recording wouldn't play. Only flags: the sound stays on the card and nothing is
    fetched here (the next admin view or visit prefetch re-fetches the sounds). True if the flag was set."""
    cached = db.get_species_info(conn, name)
    if not cached or url not in [s.get("file") for s in cached[0].get("sounds") or []]:
        return False
    data, stamp = cached
    if not data.get("sounds_broken"):
        data["sounds_broken"] = True
        db.save_species_info(conn, name, data, stamp)
        forget_cached_cards()
    return True


def _add_sounds(conn: sqlite3.Connection, name: str, data: dict, fetched_at: str) -> None:
    key = "sounds:" + name
    last = _failed_at.get(key)
    if last is not None and time.monotonic() - last < RETRY_AFTER:
        return  # it just failed; don't hammer xeno-canto
    try:
        with httpx.Client(follow_redirects=True) as http:
            data["sounds"] = fetch_sounds(http, data.get("sci_name"), data.get("map_center"))
    except Exception:
        log.exception("xeno-canto lookup failed for %s", name)
        _failed_at[key] = time.monotonic()
        return
    data.pop("sounds_broken", None)
    db.save_species_info(conn, name, data, fetched_at)
    forget_cached_cards()


def get(conn: sqlite3.Connection, name: str, fetch: bool = True) -> dict | None:
    """The cached card, built on first use and rebuilt monthly. A failed rebuild keeps the old card."""
    if not has_card(name):
        return None
    cached = db.get_species_info(conn, name)
    fresh = cached and datetime.now() - datetime.fromisoformat(cached[1]) < REFRESH_AFTER
    if fresh and fetch and _needs_sounds(cached[0]):
        _add_sounds(conn, name, cached[0], cached[1])
    if fresh or not fetch:
        return cached[0] if cached else None
    last = _failed_at.get(name)
    if last is not None and time.monotonic() - last < RETRY_AFTER:
        return cached[0] if cached else None  # it just failed; don't hammer the sources
    with _lock:
        if name in _building:  # someone else is building this card right now
            return cached[0] if cached else None
        _building.add(name)
    try:
        return _rebuild(conn, name, cached)
    finally:
        with _lock:
            _building.discard(name)


def _rebuild(conn: sqlite3.Connection, name: str, cached: tuple[dict, str] | None) -> dict | None:
    old = cached[0] if cached else None
    try:
        data = build(name)
    except Exception:
        log.exception("species card failed for %s", name)
        _failed_at[name] = time.monotonic()
        return old
    if not data.get("inat_id") and not data.get("description"):
        _failed_at[name] = time.monotonic()
        return old  # nothing found; try again later instead of caching a blank card
    stamp = datetime.now().replace(microsecond=0).isoformat()
    if old:
        # A rebuild while a source is down must not wipe what the old card had: new data wins, gaps keep the old.
        gaps = [k for k in MERGED_FIELDS if old.get(k) and not data.get(k)]
        data = {**old, **{k: v for k, v in data.items() if v not in (None, "", [], {})}}
        if gaps:
            stamp = cached[1]  # keep the old date so it is rebuilt again (after the backoff) to fill the gaps
            _failed_at[name] = time.monotonic()
    if data.get("sounds") and old and data["sounds"] != old.get("sounds"):
        data.pop("sounds_broken", None)  # fresh recordings replace the ones that wouldn't play
    db.save_species_info(conn, name, data, stamp)
    forget_cached_cards()
    return data


def _photo_of(card_photo: dict | None) -> dict | None:
    return {"url": card_photo["url"], "credit": card_photo.get("credit") or "", "license": card_photo.get("license") or ""} \
        if card_photo and card_photo.get("url") else None


def reference_photo(conn: sqlite3.Connection, name: str, fetch: bool = True) -> dict | None:
    """A small reference photo ({url, credit, license}) for a species, for comparing with the owner's own.
    The cached species card's photo if there is one, else a photo-only iNaturalist lookup cached 30 days under
    settings `refphoto:<name>` (a miss is cached too). Never builds a card or calls Haiku; with fetch=False
    it only reads caches."""
    if not has_card(name):
        return None
    name = ALIASES.get(name, name)
    card = db.get_species_info(conn, name)
    if card and (photo := _photo_of(card[0].get("photo"))):
        return photo
    key = f"refphoto:{name}"
    raw = db.get_value(conn, key)
    if raw:
        saved = json.loads(raw)
        if datetime.now() - datetime.fromisoformat(saved["at"]) < REFRESH_AFTER:
            return saved["photo"]
    if not fetch:
        return None
    photo = None
    try:
        with httpx.Client() as client:
            taxon = find_taxon(client, name)
            if taxon:
                p = open_photo(client, taxon)
                url = p.get("medium_url") or p.get("url")
                if url:
                    photo = {"url": _https(url), "credit": re.sub(r",? ?uploaded by .*$", "", p.get("attribution") or ""),
                             "license": LICENSES.get(p.get("license_code"), p.get("license_code") or "")}
    except Exception:
        log.exception("reference photo lookup failed for %s", name)
        return None  # network trouble: try again next time
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                 (key, json.dumps({"photo": photo, "at": datetime.now().replace(microsecond=0).isoformat()})))
    conn.commit()
    return photo


def prefetch(name: str) -> None:
    """Build a new species' card in the background so it's ready when the alert is opened."""
    with _lock:
        if not has_card(name) or name in _building:
            return

    def run():  # get() takes the _building guard itself
        with db.session() as conn:
            get(conn, name)

    threading.Thread(target=run, daemon=True).start()


def add_facts(conn: sqlite3.Connection, name: str, http: httpx.Client) -> bool:
    """Refresh one cached card's notes and header facts from its Wikipedia article. Only those two fields change; the
    card keeps its date. Raises if a source fails (the caller keeps the old card); False if there is nothing to read."""
    cached = db.get_species_info(conn, name)
    if not cached:
        return False
    card, fetched_at = cached
    wiki = card.get("wiki_url")
    if not wiki or not card.get("description"):
        return False
    _, article = wikipedia_article(http, unquote(wiki.rsplit("/", 1)[-1]), patient=True)
    if not article:
        return False
    text = card_text(card.get("common") or name, article)
    if not text["facts"]:
        return False
    card["facts"] = text["facts"]
    if text["notes"]:
        card["notes"] = text["notes"]
    db.save_species_info(conn, name, card, fetched_at)
    forget_cached_cards()
    return True


def backfill_facts(conn: sqlite3.Connection, redo: bool = False, only: str | None = None, dry_run: bool = False,
                   out=print, coin: bool = False) -> tuple[int, int]:
    """Give every cached species card header facts. Returns (done, failed). Cards that already have facts are skipped
    unless redo. With dry_run it only lists the cards it would do."""
    names = [r["name"] for r in conn.execute("SELECT name FROM species_info ORDER BY name")]
    todo = []
    for name in names:
        if only and name.lower() != only.lower():
            continue
        card = (db.get_species_info(conn, name) or ({}, ""))[0]
        if card.get("wiki_url") and card.get("description") and (
                redo or not card.get("facts") or (coin and vague_coin(card.get("facts")))):
            todo.append(name)
    if dry_run:
        out(f"would do {len(todo)}: {', '.join(todo)}")
        return 0, 0
    done = failed = 0
    with httpx.Client(follow_redirects=True) as http:
        for i, name in enumerate(todo, 1):
            if i > 1:
                _sleep(PACE)
            try:
                if add_facts(conn, name, http):
                    done += 1
                else:
                    failed += 1
            except Exception:
                log.exception("facts failed for %s", name)
                failed += 1
            out(f"{i}/{len(todo)} done, {failed} failed" if failed else f"{i}/{len(todo)} done, 0 failed")
    return done, failed


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m birdbrain.species_info")
    parser.add_argument("--facts", action="store_true", help="add header facts (and refresh notes) on cached cards")
    parser.add_argument("--redo", action="store_true", help="with --facts: also cards that already have facts")
    parser.add_argument("--coin", action="store_true", help="with --facts: only cards whose facts say \"a coin\" without naming it")
    parser.add_argument("--name", help="with --facts: only this species")
    parser.add_argument("--dry-run", action="store_true", help="with --facts: list the cards, change nothing")
    args = parser.parse_args()
    if not args.facts:
        parser.error("nothing to do (give --facts)")
    with db.session() as conn:
        db.init_db(conn)
        done, failed = backfill_facts(conn, args.redo, args.name, args.dry_run, coin=args.coin)
    print(f"facts added to {done} cards, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
