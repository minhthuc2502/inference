"""Prompts from a local file or a Hugging Face dataset; same interface as a reference Dataset.

`tokenize: false` keeps plain text in `input_ids` (the server tokenizes; use with backend api=chat or completions).

`path`: local .jsonl/.json/.csv/.parquet/.pkl file, or a HF dataset id (`name`, optional `subset`/`split`).
"""
import os

import pandas as pd

_READERS = {
    ".jsonl": lambda p: pd.read_json(p, lines=True),
    ".json": pd.read_json,
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".pkl": pd.read_pickle,
    ".pickle": pd.read_pickle,
}


def load_prompts(path, column, subset=None, split="train"):
    ext = os.path.splitext(path)[1].lower()
    if os.path.isfile(path):
        if ext not in _READERS:
            raise ValueError(f"unsupported file type '{ext}'; use one of {sorted(_READERS)}")
        return _READERS[ext](path)[column].tolist()
    from datasets import load_dataset

    return list(load_dataset(path, subset, split=split)[column])


class Dataset:
    def __init__(
        self,
        encode,
        path,
        column="text_input",
        subset=None,
        split="train",
        total_sample_count=None,
        tokenize=True,
        chat_template_tokenizer=None,
    ):
        prompts = load_prompts(path, column, subset, split)[:total_sample_count]
        if not tokenize:
            self.input_ids = prompts
        elif chat_template_tokenizer:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(chat_template_tokenizer)
            texts = [
                tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                for p in prompts
            ]
            self.input_ids = tok(texts, add_special_tokens=False)["input_ids"]
        else:
            self.input_ids = [encode(p) for p in prompts]
        self.total_sample_count = len(self.input_ids)
        self.perf_count = self.total_sample_count

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass
