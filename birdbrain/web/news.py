"""One-time "what's new" pop-up for everyone who opens the site (base.html renders it, static/news.js shows it).

A notice has an id and a last day. The server leaves the pop-up out after that day; before it, the browser shows it
once and remembers the dismissal in its own storage (`bb_news_<id>`), so it never comes back in that browser. The
tour's step about the same news sets the same flag (static/tour.js), so a first-time visitor gets the tour, not both.

Only one notice so far: visit clips. Clips never leave the Pi, so the public mirror never gets this notice, and a
site with clips turned off doesn't either. After CLIPS_UNTIL the notice is dead code and can be removed."""
from datetime import date

CLIPS_ID = "clips"
CLIPS_UNTIL = date(2026, 11, 25)  # two weeks from the day the notice went out (2026-11-11); the last day it shows


def clips_rules(days: int, per_species: int) -> str:
    """How long clips are kept, in one light line. Only what a visitor can make sense of: the age limit and the
    newest few of each species (clips.prune also keeps starred visits' clips, but stars are the owner's)."""
    if days <= 0:
        return "Clips are kept for good."
    if per_species <= 0:
        return f"Clips are kept for {days} days."
    return f"Clips are kept for {days} days; the newest {per_species} of each species are kept longer."


def clips_notice(today: date, *, public: bool, seconds: float, days: int, per_species: int) -> dict | None:
    """The clips notice for today's page, or None: past its last day, on the public mirror, or clips are off."""
    if public or seconds <= 0 or today > CLIPS_UNTIL:
        return None
    return {
        "id": CLIPS_ID,
        "until": CLIPS_UNTIL.isoformat(),
        "title": "Visits now come with a short clip",
        "text": f"New visits keep a {seconds:g} second video clip, without sound. "
                "Open a visit and press Play clip to watch it.",
        "rules": clips_rules(days, per_species),
        "go": "See the clips",
        "href": "/?kind=clip",
    }
