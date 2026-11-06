"""Shared helpers for the also tests (used by more than one test module)."""
from birdbrain.identify import Circle, Sighting


BUSHTITS = [{"visitor": "bird", "species": "Bushtit", "count": 2, "confidence": "high",
             "circle": {"x": 0.6, "y": 0.4, "r": 0.03, "claude": True}}]


def junco(confidence="high", also=()) -> Sighting:
    return Sighting(visitor="bird", species="Dark-eyed Junco", count=1, confidence=confidence, notes="",
                    circle=Circle(x=0.5, y=0.5, r=0.1), also=list(also))
