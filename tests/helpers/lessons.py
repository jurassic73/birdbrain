"""Shared helpers for the lessons tests (used by more than one test module)."""
import pytest

from birdbrain import lessons


@pytest.fixture(autouse=True)
def synchronous_distill(monkeypatch):
    """The automatic distill runs in the request itself, so tests never leave a thread behind."""
    monkeypatch.setattr(lessons, "BACKGROUND", False)
    monkeypatch.setattr(lessons, "_running", False)


@pytest.fixture
def fake_distill(monkeypatch):
    """Claude's answer: a short list made from the lines it was given. Records what it was asked."""
    calls = []

    def ask(lines):
        calls.append(lines)
        return lessons.Distilled(points=["Rock tips after a refill are not birds.", "- Juncos are the usual small bird."],
                                 changes="Merged three rock notes."), \
            {"model": lessons.MODEL, "input_tokens": 2000, "output_tokens": 120}
    monkeypatch.setattr(lessons, "ask", ask)
    return calls


def notes(add_visit, n, start=0):
    return [add_visit(corrected_visitor="none", feedback=f"rock tip number {i} poking out of the water after a refill")
            for i in range(start, start + n)]
