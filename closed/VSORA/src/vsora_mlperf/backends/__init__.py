import importlib

from .base import Backend

# name -> "module:Class"; imported lazily so a backend's SDK is only needed when selected.
# In-process engines (<backend>/engine.py) run the model inside the harness; servers (<backend>/server.py, or
# "openai" for an endpoint already running) are driven over the OpenAI completions API.
REGISTRY = {
    "hf": "vsora_mlperf.backends.hf.engine:HFBackend",
    "vllm": "vsora_mlperf.backends.vllm.engine:VllmBackend",
    "vllm_server": "vsora_mlperf.backends.vllm.server:VllmServerBackend",
    "openai": "vsora_mlperf.backends.openai_api.client:OpenAIBackend",
    "vsora": "vsora_mlperf.backends.vsora.engine:VsoraBackend",
}


def create_backend(name: str, **kwargs) -> Backend:
    if name not in REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Available: {sorted(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)(**kwargs)


__all__ = ["Backend", "REGISTRY", "create_backend"]
