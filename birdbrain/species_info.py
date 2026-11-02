"""Species info cards: reference photo, description, local status and seasonality, range map, behavior notes, links.

Built once per species from free sources (iNaturalist, Wikipedia, the eBird taxonomy) plus one small Claude Haiku
call, cached in the species_info table and refreshed monthly. Any source can fail; the card shows what it got.
"""

import json
import logging
import math
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from urllib.parse import quote

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
_lock = threading.Lock()


def has_card(name: str | None) -> bool:
    return bool(name) and name not in NO_CARD


def _get(client: httpx.Client, url: str, **params) -> dict:
    r = client.get(url, params=params or None, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


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


def ebird_code(client: httpx.Client, common: str) -> str | None:
    """eBird species code by common name, from the eBird taxonomy (downloaded once, ~3.5 MB, no key needed)."""
    path = config.DATA_DIR / "ebird_codes.json"
    if not path.exists():
        taxa = _get(client, "https://api.ebird.org/v2/ref/taxonomy/ebird", fmt="json", cat="species")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({t["comName"].lower(): t["speciesCode"] for t in taxa}))
    return json.loads(path.read_text()).get(common.lower())


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
        data = _get(client, "https://xeno-canto.org/api/3/recordings", query=query, key=config.XC_API_KEY, per_page=200)
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


class Notes(BaseModel):
    notes: list[str]


NOTES_PROMPT = """You write short behavior notes for a backyard wildlife site, using only the article text you are given.
Write 2 to 4 notes, each one sentence of at most 25 words, in plain language:
diet; how it behaves at a feeder or birdbath (or in a backyard); how it gets along with other animals (aggressive, territorial, social); one fun fact.
Skip any topic the article does not cover. Never add facts that are not in the text."""


def behavior_notes(name: str, article: str) -> list[str]:
    from .identify import HAIKU, client  # lazy: keeps the web app from loading the SDK at startup

    response = client().beta.messages.parse(
        model=HAIKU,
        max_tokens=1000,
        output_config={"effort": "low"},
        system=NOTES_PROMPT,
        messages=[{"role": "user", "content": f"Animal: {name}\n\nArticle:\n{article[:12000]}"}],
        output_format=Notes,
    )
    parsed = response.parsed_output
    return [n.strip() for n in parsed.notes[:4] if n.strip()] if parsed else []


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
            summary = _get(http, f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}")
            if summary.get("type") == "standard":
                data["description"] = summary.get("extract")
                wiki_url = summary["content_urls"]["desktop"]["page"]
                ext = _get(http, "https://en.wikipedia.org/w/api.php", action="query", prop="extracts",
                           explaintext=1, redirects=1, format="json", titles=summary["title"])
                article = next(iter(ext["query"]["pages"].values())).get("extract", "")
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
            data["notes"] = behavior_notes(common, article)
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


_season_cache: tuple[float, dict] = (0.0, {})


def forget_cached_cards() -> None:
    global _season_cache
    _season_cache = (0.0, {})


def cached_cards() -> dict[str, dict]:
    """Every cached card by name, re-read at most once a minute (pages call this for each visit card)."""
    global _season_cache
    read_at, cards = _season_cache
    if time.monotonic() - read_at > 60:
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


def short_status(label: str | None) -> tuple[str, str] | None:
    """(short, full) local status for a cached card, for tight lists; never the network."""
    if not has_card(label):
        return None
    full = (cached_cards().get(label) or {}).get("status")
    return (SHORT_STATUS.get(full, full), full) if full else None


def _needs_sounds(data: dict) -> bool:
    """Cards built before the xeno-canto key was set get their sounds added on next view, no full rebuild."""
    return bool(config.XC_API_KEY) and data.get("is_bird") and "sounds" not in data


def _add_sounds(conn: sqlite3.Connection, name: str, data: dict, fetched_at: str) -> None:
    try:
        with httpx.Client(follow_redirects=True) as http:
            data["sounds"] = fetch_sounds(http, data.get("sci_name"), data.get("map_center"))
    except Exception:
        log.exception("xeno-canto lookup failed for %s", name)
        return
    db.save_species_info(conn, name, data, fetched_at)


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
    try:
        data = build(name)
    except Exception:
        log.exception("species card failed for %s", name)
        return cached[0] if cached else None
    if not data.get("inat_id") and not data.get("description"):
        return cached[0] if cached else None  # nothing found; try again next time instead of caching a blank card
    db.save_species_info(conn, name, data, datetime.now().replace(microsecond=0).isoformat())
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
        _building.add(name)

    def run():
        try:
            with db.session() as conn:
                get(conn, name)
        finally:
            with _lock:
                _building.discard(name)

    threading.Thread(target=run, daemon=True).start()
