#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec bash IMTSTokenizer/scripts/launch_parallel.sh \
  --plan IMTSTokenizer/configs/final_from_scratch.json \
  --gpus 0 1 2 4 5 6 --per-gpu 2 --seeds 42 43 44 45 46 \
  --dataset-per-gpu mimic=1 "$@"
