"""Shared helpers for the mirror tests (used by more than one test module)."""

# The owner's circle checks (Ask BioCLIP 2 / Haiku / Sonnet on the visit page) go to the mirror unless Settings >
# Public mirror says not to. The setting is read on the Pi while the copy is built, so an unshared check never leaves.
CHECKED_RAW = {"calls": {"haiku": {"input_tokens": 10}, "circle": {"x": 0.5, "y": 0.5, "r": 0.1},
                         "owner_checks": [{"source": "haiku", "say": "Haiku says Bushtit (high).", "usd": 0.001,
                                           "circle": {"x": 0.4, "y": 0.4, "r": 0.05}, "at": "2026-11-03T10:00:00",
                                           "answer": {"notes": "crest and tail"}}]},
               "trigger": "rim"}
