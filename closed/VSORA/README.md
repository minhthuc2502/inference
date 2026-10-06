# VSORA MLPerf Inference

Closed-division-style layout (modelled on `closed/NVIDIA` and `closed/AMD` in
[inference_results_v6.1](https://github.com/mlcommons/inference_results_v6.1/tree/main/closed))
for bringing up the VSORA chip. Stage 1: validate the harness end-to-end on this machine with a tiny model.
Stage 2: add the VSORA backend.

## Layout: one self-contained directory per model (like `closed/AMD/src/<model>/`)
```
<model>/
  offline_<system>.yaml   # one self-contained run config per scenario and system
  server_<system>.yaml
  smoke_<system>.yaml     # synthetic plumbing test
  user_<system>.conf      # LoadGen overrides (target_qps, min_duration, ...) for this model on this system
```
Each yaml has `benchmark_name`, `scenario`, `test_mode`, `backend` + `backend_config`, `sampling_config`,
`dataset_config` (`script` + `args`), `harness_config` (user_conf_path) and optionally
`accuracy_config`. Paths in `user_conf_path`, `dataset_config.script` and `accuracy_config.script` are relative to the yaml. Any key can be overridden on the command line as `key.sub=value`.
Dataset loaders and accuracy scorers are per benchmark, not per model (like NVIDIA's `3rdparty/mlc-inference`): unmodified MLCommons
files vendored in `third_party/mlc-inference/` at the commit recorded in its `UPSTREAM`, referenced from
`accuracy_config.script`. `make models` lists what exists. First model: `smollm2-135m-instruct` (demo, CPU).

## Quick start
```bash
cd closed/VSORA
make setup                    # venv + CPU torch + loadgen build (TORCH_INDEX=.../whl/cu124 for NVIDIA GPU)
make smoke-offline            # synthetic prompts, seconds
make smoke-server
make smoke-offline ARGS="backend_config.device=cuda"   # hf backend on a machine with an NVIDIA GPU
```

## NVIDIA GPU with vLLM
Two ways to run an engine (`backend:` in the yaml, or the `backend=` override; same `backend_config` keys):
- `vllm_server` (default in `*_gpu.yaml`): launches `vllm serve` (`backend_config` keys become `--kebab-case`
  flags), then the harness is a plain OpenAI-completions client, as in NVIDIA's trtllm-serve/Dynamo harness and
  the reference gpt-oss/deepseek-r1 (SGLang over HTTP). The server does queuing and continuous batching.
- `vllm`: in-process engine (`vllm.LLM` Offline, async engine Server), as in the reference `SUT_VLLM.py`.
- `openai`: any OpenAI-compatible server already running (`backend_config.base_url`, `model`), e.g. sglang,
  trtllm-serve, or a future VSORA server.
```bash
VLLM=1 make setup             # vLLM (with its own CUDA torch) instead of CPU torch
make smoke-offline SYSTEM=gpu
make run SYSTEM=gpu SCENARIO=server
make run SYSTEM=gpu ARGS="backend=vllm"   # in-process engine instead of the server
```
Measure and update `target_qps` in `user_gpu.conf` after the first run.

## Real dataset
Like the vendor harnesses, samples are loaded by the benchmark's reference `Dataset` class
(`third_party/mlc-inference/language/<benchmark>/dataset.py`, set in `dataset_config.script`) and LoadGen's QSL is
built from its `total_sample_count`, `perf_count`, `LoadSamplesToRam`, `UnloadSamplesFromRam`.
`dataset_config.args` is passed as-is to `Dataset(...)`, so its keys follow that class's constructor.
```bash
make data                                    # official llama3.1-8b cnn_eval.json -> data/
# The official tok_input is Llama-3.1 tokens; a different dev model needs its own tokenization, once:
.venv/bin/python scripts/retokenize_dataset.py --src data/cnn_eval.json --dst data/cnn_eval_smollm2.json \
    --model HuggingFaceTB/SmolLM2-135M-Instruct --chat-template
make run                                     # MODEL=smollm2-135m-instruct SCENARIO=offline SYSTEM=cpu, ~5 min
make run SCENARIO=server                     # functional check; INVALID on this CPU (see user_cpu.conf)
make accuracy                                # accuracy run + official ROUGE script
make run ARGS="dataset_config.args.dataset_path=<file>"
```
With Llama-3.1-8B-Instruct the official `cnn_eval.json` is used as-is, no re-tokenization.

`dataset_config.args.total_sample_count` (first N rows) and `min_query_count` in `user_<system>.conf` must be changed
together: LoadGen reuses samples to reach `min_query_count`. The Offline sizing formula is in `user_cpu.conf`.

## Add a model
Copy `smollm2-135m-instruct/` to `<model>/`, change `backend_config.model`, `dataset_config`,
`benchmark_name`, then measure throughput and set `target_qps` in `user_<system>.conf`.

Results land in `results/<model>/<benchmark_name>/<scenario>/<test_mode>/` (git-ignored).
See `AGENTS.md` for architecture and how to add the VSORA backend.
