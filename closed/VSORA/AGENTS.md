# AGENTS.md — closed/VSORA

Guidance for AI agents and humans working in this directory. Keep it current when architecture changes.

## Purpose
Bring up the VSORA chip for MLPerf Inference. Layout mirrors vendor dirs in
`mlcommons/inference_results_v6.1/closed/<Vendor>` (NVIDIA, AMD): `<model>` dirs, `docker`, `docs`, `scripts`,
`src`, `systems`, `results`, plus a `Makefile`. The reference LoadGen at `../../loadgen` is used as-is.

## Architecture (data flow)
`runner` -> `Backend` (model+device) + reference `Dataset` (-> lg.ConstructQSL) -> `OfflineSUT` or `ServerSUT` (by scenario) -> LoadGen.
Offline = throughput: the whole sample set, length-sorted, goes to `Backend.generate` at once. Server = latency:
each sample is its own `Backend.submit` request, first token and completion reported per sample (TTFT/TPOT).

```
src/vsora_mlperf/
  backends/   Backend contract (base.py), REGISTRY in __init__.py, one folder per engine:
                <engine>/engine.py = in-process, <engine>/server.py = launches the engine's OpenAI server.
              hf/engine.py: CPU / NVIDIA reference; no scheduler, so it queues and statically batches itself.
              vllm/engine.py: in-process vLLM (LLM Offline, async engine Server); vLLM batches itself.
              vllm/server.py: `vllm serve` + openai_api client. openai_api/client.py: any OpenAI-compatible
              endpoint (server owns queuing + continuous batching; client caps in-flight requests).
              vsora/engine.py: chip stub. aio.py: background event loop. Backends know nothing about LoadGen.
  harness/    config.py (yaml + dotted overrides), offline_sut.py / server_sut.py (LoadGen callbacks),
              responses.py (lg.QuerySamplesComplete / lg.FirstTokenComplete),
              runner.py (CLI, settings, logs), accuracy.py (official scorer from `accuracy_config:`).
  datasets/   create_dataset: imports `dataset_config.script` (the benchmark's reference dataset.py) and calls
              `Dataset(**dataset_config.args)`; row order = qsl_idx. synthetic.py = same interface, smoke only
              (no script set).
<model>/      Self-contained, as closed/AMD/src/<model>/: <scenario>_<system>.yaml (run config),
              user_<system>.conf (LoadGen overrides).
              Paths in the yaml (user_conf_path, dataset_config.script, accuracy_config.script) are relative to the yaml.
systems/      System description JSON per tested machine (submission metadata).
scripts/      setup_env.sh, download_cnndm.sh, retokenize_dataset.py. Makefile targets stay thin wrappers over these.
third_party/  mlc-inference/: unmodified MLCommons reference files (dataset loaders, accuracy scorers), same relative paths as
              upstream, pinned commit in UPSTREAM. Shared by every model of a benchmark.
results/      Run output, git-ignored.
```

## Rules
- Backend = model execution only. No `mlperf_loadgen` import in `backends/`; LoadGen stays in `harness/`.
- Contract: `load(scenario)` (picks the engine type if it matters), `encode(text)->ids` (synthetic smoke only),
  `generate(all_ids, max_new_tokens)->new tokens per prompt, EOS stripped` (Offline; backend batches as it likes),
  `submit(ids, max_new_tokens, on_complete, on_first_token=None)` (Server; returns at once, callbacks from any
  thread), `close()`. `on_first_token(int)` fires once per request as soon as its first token exists and must
  equal its final output[0] (TEST06). hf batches waiting requests on a worker thread; vllm / openai stream per request.
- Device/engine knobs (batch_size, tensor_parallel_size, gpu_memory_utilization, ...) live in `backend_config`,
  not `harness_config`.
- New backend: add `backends/<engine>/engine.py` and/or `server.py`, register lazily in `REGISTRY` ("module:Class") so its SDK is only
  imported when selected. Select via `backend:` in the yaml or the `backend=` override.
- Don't modify `../../loadgen` or top-level `mlperf.conf`; LoadGen overrides go in `<model>/user_<system>.conf`.
- Config-driven: no hard-coded model names, paths or QPS in Python.
- Real MLPerf benchmarks (llama, etc.) need their official datasets and accuracy scripts: vendor `dataset.py` and
  the scorer unmodified into `third_party/mlc-inference/<same path as upstream>` (update UPSTREAM; never edit those
  files, MLPerf requires the reference scorer), point `dataset_config.script` / `accuracy_config.script` at them,
  put the reference Dataset's constructor kwargs (dataset_path, total_sample_count, ...) in `dataset_config.args`,
  add a config, don't fake results. No VSORA-specific loaders: tokenization is offline preprocessing
  (scripts/retokenize_dataset.py for non-benchmark dev models).
- Keep code small: no comments except non-obvious WHY; no speculative abstractions.

## Add the VSORA chip (stage 2)
1. Implement `backends/vsora/engine.py` (`VsoraBackend`) against the VSORA runtime, or, if the runtime serves
   an OpenAI-compatible API, use `backend: openai` / add `vsora/server.py` like `vllm/server.py`.
2. Add `<model>/<scenario>_<vsora-system>.yaml` with `backend: vsora` + `backend_config`, and
   `user_<vsora-system>.conf` with measured `target_qps`.
3. Add `systems/<vsora-system>.json` and `docker/` for the runtime environment.
4. Validate with `make smoke-offline SYSTEM=<vsora-system>`, then compare against the `hf`/`vllm` output tokens.

## Status
- Smoke path (`hf` backend, tiny model, Offline+Server) — see README for commands.
- `vllm`, `vllm_server`, `openai` backends + `*_gpu.yaml` / `user_gpu.conf`: written against vLLM's public API
  (LLM, AsyncLLMEngine, `vllm serve` /v1/completions with `return_token_ids`) but only exercised with a stub
  engine / fake HTTP server so far; `target_qps` in user_gpu.conf are unmeasured starting points.
- Models: `smollm2-135m-instruct` (demo; CPU). CNN/DailyMail (llama3.1-8b data) + official ROUGE scoring,
  Offline VALID; Server runs but is INVALID on CPU (early stopping needs ~460 queries).
  Benchmark name deliberately not `llama3_1-8b`: that name pulls mlperf.conf rules (13368 queries,
  use_token_latencies, TTFT/TPOT) the harness can't meet yet.
- Token metrics: `use_token_latencies = 1`; Offline reports tokens/s, Server reports TTFT/TPOT via
  lg.FirstTokenComplete (first-token consistency checked TEST06-style on smoke-tiny: 32/32).
- Not yet: Llama-3.1-8B run,
  multi-device sharding, compliance tests, submission packaging.
