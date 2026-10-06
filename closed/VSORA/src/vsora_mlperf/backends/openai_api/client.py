"""Client for any OpenAI-compatible completions server (vllm serve, sglang, trtllm-serve, Dynamo, ...).

This is how recent vendor harnesses drive their engines (NVIDIA: trtllm-serve / Dynamo over OpenAI API; the
reference gpt-oss / deepseek-r1: SGLang over HTTP). The server owns queuing and continuous batching; the client
only keeps up to `max_concurrency` requests in flight. Prompts go as token ids (`/v1/completions` accepts them);
output token ids come back via `return_token_ids` (vLLM >= 0.10.2), else the text is re-tokenized like NVIDIA's
client does, which is not always token-exact.

`api: chat` posts to /v1/chat/completions: a prompt given as plain text becomes one user message and the server applies
the chat template and tokenizes. `api: completions` (default) accepts token ids or plain text. `sampling` replaces the
default greedy parameters (e.g. {} = server defaults, {min_tokens: 1}).
"""
import asyncio
import json
import os
import traceback

from ..aio import LoopThread
from ..base import Backend


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000",
        tokenizer: str | None = None,
        max_concurrency: int = 1024,
        timeout_s: float = 3600,
        api: str = "completions",
        sampling: dict | None = None,
        **_,
    ):
        if api not in ("completions", "chat"):
            raise ValueError("api must be 'completions' or 'chat'")
        self.api = api
        self.sampling = sampling
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.tokenizer_name = tokenizer or model
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self.tokenizer = None
        self._loop = None
        self._client = None
        self._slots = None

    def load(self, scenario: str) -> None:
        import httpx
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self._loop = LoopThread()

        async def create():
            limits = httpx.Limits(max_connections=self.max_concurrency, max_keepalive_connections=self.max_concurrency)
            return httpx.AsyncClient(base_url=self.base_url, limits=limits, timeout=self.timeout_s), asyncio.Semaphore(
                self.max_concurrency
            )

        self._client, self._slots = self._loop.run(create())

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text)["input_ids"]

    def _payload(self, prompt, max_new_tokens, stream):
        # Default greedy, as the reference: temperature 0, top_k 1, min_tokens 1 (first token is never EOS).
        sampling = self.sampling
        if sampling is None:
            sampling = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_tokens": 1, "seed": 42}
        payload = {"model": self.model_name, "stream": stream, "return_token_ids": True, **sampling}
        if self.api == "chat":
            if not isinstance(prompt, str):
                raise ValueError("api=chat needs plain-text prompts (dataset_config.args.tokenize: false)")
            payload["messages"] = [{"role": "user", "content": prompt}]
            payload["max_completion_tokens"] = max_new_tokens
        else:
            payload["prompt"] = prompt
            payload["max_tokens"] = max_new_tokens
        return payload

    @property
    def _route(self):
        return "/v1/chat/completions" if self.api == "chat" else "/v1/completions"

    def _text(self, choice, stream):
        if self.api == "completions":
            return choice.get("text") or ""
        return (choice.get("delta" if stream else "message") or {}).get("content") or ""

    def _tokens(self, choice):
        ids = choice.get("token_ids")
        if ids is None:
            ids = self.tokenizer(self._text(choice, False), add_special_tokens=False)["input_ids"]
        return list(ids)

    def _strip_eos(self, tokens):
        if tokens and tokens[-1] == self.tokenizer.eos_token_id:
            tokens.pop()
        return tokens

    def generate(self, batch_input_ids, max_new_tokens):
        async def one(ids):
            async with self._slots:
                r = await self._client.post(self._route, json=self._payload(ids, max_new_tokens, stream=False))
                r.raise_for_status()
                return self._strip_eos(self._tokens(r.json()["choices"][0]))

        async def all_():
            return await asyncio.gather(*(one(ids) for ids in batch_input_ids))

        return self._loop.run(all_())

    def submit(self, input_ids, max_new_tokens, on_complete, on_first_token=None):
        self._loop.spawn(self._stream(input_ids, max_new_tokens, on_complete, on_first_token))

    async def _stream(self, input_ids, max_new_tokens, on_complete, on_first_token):
        try:
            async with self._slots:
                tokens, text = [], ""
                payload = self._payload(input_ids, max_new_tokens, stream=True)
                async with self._client.stream("POST", self._route, json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        choices = json.loads(line[len("data:"):])["choices"]
                        if not choices:
                            continue
                        new = choices[0].get("token_ids")
                        if new is None:  # server without return_token_ids: re-tokenize at the end
                            text += self._text(choices[0], True)
                            new = self.tokenizer(text, add_special_tokens=False)["input_ids"][len(tokens):]
                        if on_first_token and not tokens and new:
                            on_first_token(new[0])
                        tokens += new
            on_complete(self._strip_eos(tokens))
        except BaseException:
            # LoadGen would otherwise wait forever for this sample.
            traceback.print_exc()
            os._exit(1)

    def close(self) -> None:
        if self._client is not None:
            self._loop.run(self._client.aclose())
            self._loop.stop()
        self._client = None
