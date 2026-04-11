#!/usr/bin/env bash
# Train Llama-3-Instruct (meta-llama/Meta-Llama-3-8B-Instruct) with QLoRA SimPO.
set -euo pipefail
python -m src.train_simpo --config configs/llama3_simpo_config.yaml "$@"
