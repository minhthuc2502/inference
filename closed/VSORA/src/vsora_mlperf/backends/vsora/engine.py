"""VSORA chip backend (TODO). Implement the Backend contract using the VSORA runtime/SDK.

If the VSORA runtime ships an OpenAI-compatible server instead, use `backend: openai` (or add vsora/server.py
like vllm/server.py) rather than this in-process engine."""
from ..base import Backend


class VsoraBackend(Backend):
    name = "vsora"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load(self, scenario: str) -> None:
        raise NotImplementedError("VSORA runtime integration not implemented yet")

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def generate(self, batch_input_ids, max_new_tokens):
        raise NotImplementedError

    def submit(self, input_ids, max_new_tokens, on_complete, on_first_token=None):
        raise NotImplementedError
