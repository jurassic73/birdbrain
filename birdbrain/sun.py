"""Local sunrise and sunset, worked out offline with NOAA's solar position equations (no network, no key).

Needs the site's SITE_LAT and SITE_LON in .env; without them sunrise() and sunset() return None.
"""

import math
from datetime import date, datetime, timedelta, timezone

from . import config


def _event(day: date, rising: bool, lat: float, lon: float) -> datetime | None:
    """Local (naive, the Pi's time zone) time of sunrise or sunset on day, None in polar day or night."""
    n = day.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1 + 0.5)  # fractional year at noon
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    phi = math.radians(lat)
    cos_ha = math.cos(math.radians(90.833)) / (math.cos(phi) * math.cos(decl)) - math.tan(phi) * math.tan(decl)
    if not -1 <= cos_ha <= 1:
        return None
    ha = math.degrees(math.acos(cos_ha)) * (1 if rising else -1)
    utc_minutes = 720 - 4 * (lon + ha) - eqtime
    utc = datetime(day.year, day.month, day.day) + timedelta(minutes=utc_minutes)
    local = utc.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
    return local.replace(second=0, microsecond=0)


def located() -> bool:
    return config.SITE_LAT is not None and config.SITE_LON is not None


def sunrise(day: date) -> datetime | None:
    return _event(day, True, config.SITE_LAT, config.SITE_LON) if located() else None


def sunset(day: date) -> datetime | None:
    return _event(day, False, config.SITE_LAT, config.SITE_LON) if located() else None
