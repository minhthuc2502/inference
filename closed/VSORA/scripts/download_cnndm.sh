#!/usr/bin/env bash
# Official llama3.1-8b CNN/DailyMail eval set (13368 samples, ~255 MB), same file the r2-downloader fetches.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://inference.mlcommons-storage.org/llama3.1_8b/datasets/cnn_eval.json"

mkdir -p "$HERE/data"
if [ -s "$HERE/data/cnn_eval.json" ]; then
  echo "data/cnn_eval.json already present"
else
  curl -fSL -o "$HERE/data/cnn_eval.json" "$URL"
fi
