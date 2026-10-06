"""vLLM in-process engine for NVIDIA GPUs (same engine calls as the MLCommons reference SUT_VLLM.py).

vLLM schedules and batches itself (continuous batching), so the harness only hands requests over: Offline gives
`vllm.LLM` the whole sample set; Server streams one request per sample through the async engine on a background
event loop. Extra `backend_config` keys go to the engine as-is (tensor_parallel_size, gpu_memory_utilization,
max_model_len, max_num_seqs, max_num_batched_tokens, ...).
"""
import itertools
import os
import traceback

from ..aio import LoopThread
from ..base import Backend


class VllmBackend(Backend):
    name = "vllm"

    def __init__(self, model: str, dtype: str = "auto", **engine_args):
        self.model_name = model
        self.engine_args = {"model": model, "dtype": dtype, **engine_args}
        self.tokenizer = None
        self.llm = None
        self.engine = None
        self._loop = None
        self._request_ids = itertools.count()

    def load(self, scenario: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if scenario == "Offline":
            from vllm import LLM

            self.llm = LLM(**self.engine_args)
            return
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        async def create():  # the async engine binds to the loop it is created on
            return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.engine_args))

        self._loop = LoopThread()
        self.engine = self._loop.run(create())

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text)["input_ids"]

    def _sampling_params(self, max_new_tokens, **kwargs):
        from vllm import SamplingParams

        # Greedy, as the reference: temperature 0, top_k 1, min_tokens 1 (first token is never EOS).
        return SamplingParams(temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=max_new_tokens, min_tokens=1, **kwargs)

    def _strip_eos(self, tokens):
        tokens = list(tokens)
        if tokens and tokens[-1] == self.tokenizer.eos_token_id:
            tokens.pop()
        return tokens

    def generate(self, batch_input_ids, max_new_tokens):
        from vllm.inputs import TokensPrompt

        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=ids) for ids in batch_input_ids],
            self._sampling_params(max_new_tokens),
            use_tqdm=False,
        )
        return [self._strip_eos(o.outputs[0].token_ids) for o in outputs]

    def submit(self, input_ids, max_new_tokens, on_complete, on_first_token=None):
        self._loop.spawn(self._stream(input_ids, max_new_tokens, on_complete, on_first_token))

    async def _stream(self, input_ids, max_new_tokens, on_complete, on_first_token):
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        try:
            tokens = []
            async for out in self.engine.generate(
                TokensPrompt(prompt_token_ids=input_ids),
                self._sampling_params(max_new_tokens, output_kind=RequestOutputKind.DELTA),
                request_id=str(next(self._request_ids)),
            ):
                new = out.outputs[0].token_ids
                if on_first_token and not tokens and new:
                    on_first_token(new[0])
                tokens += new
            on_complete(self._strip_eos(tokens))
        except BaseException:
            # LoadGen would otherwise wait forever for this sample.
            traceback.print_exc()
            os._exit(1)

    def close(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self._loop.stop()
        self.llm = self.engine = None
