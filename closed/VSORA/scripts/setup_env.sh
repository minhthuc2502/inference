#!/usr/bin/env bash
# Create .venv, install torch (CPU by default) + deps, build the in-repo loadgen python module.
# VLLM=1: install vLLM instead (it pins its own CUDA torch) for the vllm backend on NVIDIA GPUs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"   # set to .../whl/cu124 for NVIDIA GPU

cd "$HERE"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
if [[ "${VLLM:-0}" == 1 ]]; then
  uv pip install vllm
else
  uv pip install --index-url "$TORCH_INDEX" torch
fi
uv pip install "transformers>=4.56" numpy pybind11 setuptools wheel pyyaml
uv pip install pandas evaluate nltk rouge_score   # official llama3.1-8b accuracy script
uv pip install -e .
uv pip install --no-build-isolation "$REPO_ROOT/loadgen"
python -c "import mlperf_loadgen as lg, torch; print('loadgen OK; torch', torch.__version__, 'cuda', torch.cuda.is_available())"
