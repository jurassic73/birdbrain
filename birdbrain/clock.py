"""The site's calendar day.

Visit times are stored in the Pi's local time. A public mirror on a server in another zone (the mirror's
container runs UTC) gets the Pi's zone in its settings table as `site_timezone`, written by
scripts/push_public.sh, so "today" there matches the birdbath's today. Without that setting it's the
machine's own date, as before."""
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

KEY = "site_timezone"
RECHECK = 60  # seconds between looks at the setting
_zone: dict = {"tz": None, "checked": 0.0}


def set_zone(name: str | None) -> None:
    try:
        _zone["tz"] = ZoneInfo(name) if name else None
    except Exception:
        _zone["tz"] = None
    _zone["checked"] = time.time()


def stale() -> bool:
    return time.time() - _zone["checked"] >= RECHECK


def today() -> date:
    tz = _zone["tz"]
    return datetime.now(tz).date() if tz else date.today()


def now() -> datetime:
    """The site's local time now, naive like visits.captured_at."""
    tz = _zone["tz"]
    return datetime.now(tz).replace(tzinfo=None) if tz else datetime.now()
