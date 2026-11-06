"""Stand-ins for the Anthropic client, so identify.ask() is exercised without a network or a key."""
from birdbrain.identify import HAIKU


class FakeClient:
    """client() stand-in whose parse() returns a scripted response or raises; keeps the request it got."""

    def __init__(self, response=None, raises=None):
        self.beta = self
        self.messages = self
        self.response, self.raises = response, raises
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        if self.raises:
            raise self.raises
        return self.response


class FakeResponse:
    def __init__(self, stop_reason, parsed_output=None, model=HAIKU, **usage):
        self.model, self.stop_reason, self.parsed_output = model, stop_reason, parsed_output
        self.usage = type("Usage", (), {"input_tokens": 1200, "output_tokens": 40, **usage})()
