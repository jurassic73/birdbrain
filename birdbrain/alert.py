"""Push notifications via ntfy.sh.

Fires when:
  - A species is seen for the first time ("New species: Dark-eyed Junco!")
  - A raccoon or possum visits
  - A visitor shows up outside its usual local months ("Early arrival: Rufous Hummingbird")

Settings live in the database (ntfy_topic, ntfy_new_species, ntfy_mammals,
ntfy_quiet_start, ntfy_quiet_end) and are edited on the Settings page.
"""

import logging
import sqlite3
from datetime import datetime, time as dtime

import httpx

from . import db

log = logging.getLogger("birdbrain.alert")

NTFY_BASE = "https://ntfy.sh/"


def _in_quiet(t: dtime, start: str, end: str) -> bool:
    """True if t falls in the quiet window. Handles midnight crossing (e.g. 22:00-07:00)."""
    try:
        qs = dtime.fromisoformat(start)
        qe = dtime.fromisoformat(end)
    except (ValueError, TypeError, AttributeError):
        return False
    if qs <= qe:
        return qs <= t < qe
    # Crosses midnight
    return t >= qs or t < qe


def reason(conn: sqlite3.Connection, visitor: str, label: str, captured_at: datetime) -> str | None:
    """Return an alert reason for this visit, or None.

    Call BEFORE inserting the visit so the new-species check is accurate.
    """
    topic = db.get_value(conn, "ntfy_topic") or ""
    if not topic.strip():
        return None

    qs = db.get_value(conn, "ntfy_quiet_start") or ""
    qe = db.get_value(conn, "ntfy_quiet_end") or ""
    if qs and qe and _in_quiet(captured_at.time(), qs, qe):
        return None

    if visitor in ("raccoon", "possum"):
        if db.get_value(conn, "ntfy_mammals") != "0":
            return f"{visitor.capitalize()} at the birdbath!"

    if visitor == "bird" and label:
        if db.get_value(conn, "ntfy_new_species") != "0":
            # Seen before as the main animal or as another one in a photo.
            if db.count_visits(conn, species=label) == 0:
                return f"New species: {label}!"

    if visitor != "none" and db.get_value(conn, "ntfy_unusual") != "0":
        from . import species_info  # lazy: avoids a circular import at module load

        cached = db.get_species_info(conn, label) if species_info.has_card(label) else None
        badge = species_info.season_badge(cached[0] if cached else None, captured_at.month)
        if badge in ("Early arrival", "Late departure", "Unusual this time of year"):
            return f"{badge}: {label}"

    return None


def send(topic: str, title: str, message: str, image: bytes | None, click_url: str | None) -> None:
    """POST a notification to ntfy.sh. Logs on failure, never raises."""
    url = NTFY_BASE + topic
    headers: dict[str, str] = {
        "Title": title,
        "Message": message,
        "Priority": "default",
        "Tags": "raccoon" if "raccoon" in title.lower() else ("eyes" if "possum" in title.lower() else "bird"),
    }
    if click_url:
        headers["Click"] = click_url
    try:
        with httpx.Client(timeout=10) as client:
            if image:
                resp = client.post(url, content=image, headers={"Content-Type": "image/jpeg", **headers})
            else:
                resp = client.post(url, headers=headers)
        if not resp.is_success:
            log.warning("ntfy: %d %s", resp.status_code, resp.text[:200])
        else:
            log.info("ntfy alert sent: %s", title)
    except Exception:
        log.exception("ntfy send failed")


def get_settings(conn: sqlite3.Connection) -> dict:
    """All ntfy settings as a dict for the Settings page template."""
    return {
        "topic": db.get_value(conn, "ntfy_topic") or "",
        "new_species": db.get_value(conn, "ntfy_new_species") != "0",
        "mammals": db.get_value(conn, "ntfy_mammals") != "0",
        "unusual": db.get_value(conn, "ntfy_unusual") != "0",
        "quiet_start": db.get_value(conn, "ntfy_quiet_start") or "",
        "quiet_end": db.get_value(conn, "ntfy_quiet_end") or "",
    }


def save_settings(conn: sqlite3.Connection, *, topic: str, new_species: bool, mammals: bool,
                  quiet_start: str, quiet_end: str, unusual: bool = True) -> None:
    db.set_value(conn, "ntfy_topic", topic.strip())
    db.set_value(conn, "ntfy_new_species", "1" if new_species else "0")
    db.set_value(conn, "ntfy_mammals", "1" if mammals else "0")
    db.set_value(conn, "ntfy_unusual", "1" if unusual else "0")
    db.set_value(conn, "ntfy_quiet_start", quiet_start.strip())
    db.set_value(conn, "ntfy_quiet_end", quiet_end.strip())
