"""Shared helpers for the species info tests (used by more than one test module)."""
import pytest

from helpers.cards import CARD, SOUNDS
from helpers.clock import FROZEN_NOW
from birdbrain import db, species_info


@pytest.fixture
def fake_build(monkeypatch):
    calls = []

    def build(name):
        calls.append(name)
        return {**CARD, "name": name}
    monkeypatch.setattr(species_info, "build", build)
    return calls


HOME = [45.78, -122.48]


SUMMER = {"local_counts": [0, 0, 0, 20, 40, 50, 50, 40, 20, 0, 0, 0], "expected_months": [4, 5, 6, 7, 8, 9]}


def _save_card(conn, **extra):
    db.save_species_info(conn, "Bushtit", {**CARD, "sounds": [dict(s) for s in SOUNDS], **extra},
                         FROZEN_NOW.isoformat())


class FakeHaiku:
    """identify.client() stand-in: parse() returns the scripted CardText and keeps the request."""

    def __init__(self, notes, facts):
        self.beta = self
        self.messages = self
        self.parsed = species_info.CardText(notes=notes, facts=facts)
        self.requests = []

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        return type("R", (), {"parsed_output": self.parsed})()


def F(lead, text):
    return {"lead": lead, "text": text}
