"""Hugging Face transformers backend. Runs on CPU or NVIDIA GPU (device=cuda); reference for other backends.

Static batching only: Offline runs `batch_size` prompts at a time, Server batches whatever requests are waiting
(up to `batch_size`) on one worker thread.
"""
import os
import queue
import threading
import traceback

from ..base import Backend


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model: str, device: str = "cpu", dtype: str = "float32", batch_size: int = 1, **_):
        self.model_name = model
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self._requests = queue.Queue()
        self._worker = None

    def load(self, scenario: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(self.model_name, dtype=getattr(torch, self.dtype))
            .to(self.device)
            .eval()
        )
        if scenario == "Server":
            self._worker = threading.Thread(target=self._serve, daemon=True)
            self._worker.start()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text)["input_ids"]

    def generate(self, batch_input_ids, max_new_tokens):
        results = []
        for i in range(0, len(batch_input_ids), self.batch_size):
            results += self._generate_batch(batch_input_ids[i : i + self.batch_size], max_new_tokens)
        return results

    def submit(self, input_ids, max_new_tokens, on_complete, on_first_token=None):
        self._requests.put((input_ids, max_new_tokens, on_complete, on_first_token))

    def _serve(self):
        while True:
            batch = [self._requests.get()]
            while len(batch) < self.batch_size and not self._requests.empty():
                batch.append(self._requests.get_nowait())
            if batch[-1] is None:
                return
            try:
                ids, max_new, on_complete, on_first = zip(*batch)
                first = [f for f in on_first if f]
                outputs = self._generate_batch(
                    list(ids),
                    max(max_new),
                    on_first_token=(lambda toks: [f(t) for f, t in zip(on_first, toks) if f]) if first else None,
                )
                for done, out in zip(on_complete, outputs):
                    done(out)
            except BaseException:
                # LoadGen would otherwise wait forever for these samples.
                traceback.print_exc()
                os._exit(1)

    def _generate_batch(self, batch_input_ids, max_new_tokens, on_first_token=None):
        import torch

        batch = self.tokenizer.pad({"input_ids": batch_input_ids}, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,  # matches the MLPerf reference (min_tokens=1); first token is never EOS
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                streamer=_FirstTokenStreamer(on_first_token) if on_first_token else None,
            )
        eos = self.tokenizer.eos_token_id
        results = []
        for row in out[:, batch["input_ids"].shape[1]:].tolist():
            if eos in row:
                row = row[: row.index(eos)]
            results.append(row)
        return results

    def close(self) -> None:
        if self._worker:
            self._requests.put(None)
            self._worker.join()
        self.model = None


class _FirstTokenStreamer:
    """transformers streamer: generate() puts the prompt first, then one (batch,) tensor per decode step."""

    def __init__(self, callback):
        self._callback = callback
        self._puts = 0

    def put(self, value):
        self._puts += 1
        if self._puts == 2:
            self._callback(value.tolist())

    def end(self):
        pass
