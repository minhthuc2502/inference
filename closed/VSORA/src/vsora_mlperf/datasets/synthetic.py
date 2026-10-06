"""Deterministic synthetic prompts for plumbing/smoke tests (not an MLPerf dataset), same interface as a reference Dataset."""
import random

_WORDS = (
    "chip memory tensor kernel latency throughput server offline stream batch "
    "token model cache bandwidth compute sparse dense quantize attention layer "
    "queue thread device driver firmware compiler graph schedule pipeline"
).split()


class Dataset:
    def __init__(self, encode, total_sample_count, words_per_prompt, seed):
        rng = random.Random(seed)
        self.input_ids = [
            encode("Continue the text: " + " ".join(rng.choices(_WORDS, k=words_per_prompt)))
            for _ in range(total_sample_count)
        ]
        self.total_sample_count = total_sample_count
        self.perf_count = total_sample_count

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass
