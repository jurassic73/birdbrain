"""Shared helpers for the other visits tests (used by more than one test module)."""
import pytest

from helpers.clock import FROZEN_NOW


TODAY = FROZEN_NOW.date().isoformat()


def other(add_visit, **fields) -> int:
    return add_visit(**{"visitor": "other", "species": "Domestic Cat", "confidence": "high",
                        "captured_at": f"{TODAY}T11:00:00", "image_path": "2026/11/01/cat.jpg",
                        "thumb_path": "2026/11/01/cat_t.jpg", **fields})


@pytest.fixture
def yard(add_visit):
    """A bird, then a newer "other" (the newest visit of all)."""
    bird = add_visit(captured_at=f"{TODAY}T10:00:00", species="Bushtit")
    cat = other(add_visit)
    return bird, cat
