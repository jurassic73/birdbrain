"""Shared helpers for the web tests (used by more than one test module)."""
import re
from datetime import datetime

import pytest

from birdbrain import config, db


def _card(conn, name, **fields):
    db.save_species_info(conn, name, {"name": name, "links": [], **fields}, datetime.now().isoformat())


def more_link(html: str) -> str:
    """The Load more button's URL in a page or fragment."""
    import html as htmllib
    m = re.search(r'class="load-more" hx-get="([^"]+)"', html)
    assert m, "no Load more button"
    return htmllib.unescape(m.group(1))


def _seed_swipe_species(add_visit):
    # visits order: Bushtit (3), Song Sparrow (2), Anna's Hummingbird (1); A to Z: Anna's, Bushtit, Song
    for _ in range(3):
        add_visit(species="Bushtit", captured_at="2026-10-01T09:00:00")
    for _ in range(2):
        add_visit(species="Song Sparrow", captured_at="2026-10-05T09:00:00")
    add_visit(species="Anna's Hummingbird", captured_at="2026-10-09T09:00:00")


@pytest.fixture
def calibrated(monkeypatch):
    monkeypatch.setattr(config, "MOTION_RING", (0.5, 0.5, 0.25, 0.3, 0.2, 0.22, 0.48))
    monkeypatch.setattr(config, "MOTION_SPOTS", ((0.5, 0.45, 0.04, 0.04),))
    monkeypatch.setattr(config, "CROP_ROI", (0.2, 0.1, 0.6, 0.8))
    monkeypatch.setattr(config, "MOTION_MIN_AREA", 0.02)


def card_ids(html: str) -> list[int]:
    return [int(i) for i in re.findall(r'class="card" href="/visit/(\d+)', html)]


FOUR = ("Motion detector", "Pi animal check", "BioCLIP 2 species check", "Claude")


def _keys_dialog(html):
    """The text of the Keyboard shortcuts pop-up's rows: [(key, what it does)] and its headings."""
    dlg = html[html.index('id="keys-dialog"'):]
    dlg = dlg[:dlg.index("</dialog>")]
    rows = re.findall(r"<li><kbd>(.*?)</kbd><span>(.*?)</span></li>", dlg)
    heads = re.findall(r'<li class="keys-head">(.*?)</li>', dlg)
    return rows, heads
