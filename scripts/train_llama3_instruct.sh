#!/usr/bin/env bash
# Train Llama-3-Instruct (meta-llama/Meta-Llama-3-8B-Instruct) with QLoRA DPO.
set -euo pipefail
python -m src.train_dpo --config configs/dpo_qlora_llama3_8b.yaml "$@"
