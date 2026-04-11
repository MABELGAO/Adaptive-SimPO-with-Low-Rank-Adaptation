#!/bin/bash
set -e

echo "============================================================"
echo "  Starting SimPO Alignment Pipeline for Llama-3-8B-Instruct "
echo "============================================================"

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

LOG_DIR="./simpo_metrics_output"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/training_terminal_$(date +%Y%m%d_%H%M%S).log"

echo "[*] GPU Device initialized: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[*] Logging terminal output to: ${LOG_FILE}"
echo "[*] Executing Python script: run_simpo_llama3.py..."
echo "------------------------------------------------------------"

python run_simpo_llama3_v3.py 2>&1 | tee ${LOG_FILE}

echo "------------------------------------------------------------"
echo "[*] Training Pipeline Completed Successfully!"
echo "[*] Formal Model saved in: ./llama-3-8b-instruct-simpo-final"
echo "[*] Visualizations & Metrics saved in: ${LOG_DIR}"
echo "============================================================"
