"""Samples come from the benchmark's reference `Dataset` class (yaml `dataset_config.script`), as vendor harnesses do.
`dataset_config.type: text` loads prompts from a local file or HF dataset (datasets/text.py).

Interface used by the harness, shared by every MLCommons language/<benchmark>/dataset.py: `input_ids`,
`total_sample_count`, `perf_count`, `LoadSamplesToRam`, `UnloadSamplesFromRam`. No script and no type = synthetic smoke data.
"""
import importlib.util
import pathlib

from . import synthetic, text


def create_dataset(dataset_cfg: dict, encode):
    args = dataset_cfg.get("args", {})
    script = dataset_cfg.get("script")
    if script is None:
        module = {"text": text, None: synthetic}[dataset_cfg.get("type")]
        return module.Dataset(encode, **args)
    path = pathlib.Path(script)
    spec = importlib.util.spec_from_file_location(f"mlc_{path.parent.name}_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Dataset(**args)


__all__ = ["create_dataset"]
