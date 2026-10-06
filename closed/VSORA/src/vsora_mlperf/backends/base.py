"""Backend contract. A backend owns the model and device; it knows nothing about LoadGen.

Two entry points because the scenarios optimise different things: `generate` (Offline, throughput over the whole
sample set) and `submit` (Server, per-request latency: TTFT/TPOT).
"""
from abc import ABC, abstractmethod
from collections.abc import Callable


class Backend(ABC):
    name = "base"

    @abstractmethod
    def load(self, scenario: str) -> None:
        """Load weights/tokenizer onto the device. `scenario` ("Offline"/"Server") picks the engine type if it matters."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Prompt text -> token ids."""

    @abstractmethod
    def generate(self, batch_input_ids: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        """Offline: greedy-generate every prompt, batching however is fastest for the device.

        Returns only the new tokens per prompt, in input order, EOS stripped.
        """

    @abstractmethod
    def submit(
        self,
        input_ids: list[int],
        max_new_tokens: int,
        on_complete: Callable[[list[int]], None],
        on_first_token: Callable[[int], None] | None = None,
    ) -> None:
        """Server: start one greedy request and return immediately; callbacks may run on any thread.

        `on_first_token(token)` fires as soon as the first new token exists (it must equal the final output[0],
        TEST06); `on_complete(tokens)` fires with the new tokens, EOS stripped.
        """

    def close(self) -> None:
        """Release device resources."""
