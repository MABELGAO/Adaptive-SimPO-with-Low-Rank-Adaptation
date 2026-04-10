# Llama-3 8B SimPO Alignment Pipeline

An end-to-end Machine Learning Operations (MLOps) pipeline for human preference alignment of the `Meta-Llama-3-8B-Instruct` model. This repository implements **SimPO** (Simple Preference Optimization) via **QLoRA** to fine-tune the model efficiently on consumer-grade or standard cloud GPUs.

## Overview

This project provides a highly structured, automated pipeline designed to optimize LLMs without the need for a reference model (unlike standard DPO). It uses the `argilla/dpo-mix-7k` preference dataset and is split into two rigorous engineering phases:
1. **Heuristic Hyperparameter Tuning:** A rapid grid search over learning rates and SimPO margin parameters ($\gamma$).
2. **Formal Preference Training:** Full-epoch alignment using the optimal parameters, complete with automated metrics logging and Matplotlib visualizations.

## Pipeline Architecture

* **Base Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
* **Dataset:** `argilla/dpo-mix-7k` (90% Train / 10% Test split)
* **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes`
* **Parameter-Efficient Fine-Tuning:** LoRA ($r=16, \alpha=32$, targeting all linear layers)
* **Alignment Algorithm:** SimPO (`trl.experimental.cpo.CPOTrainer` with `cpo_alpha=0.0`)

## Installation & Requirements

Ensure you have a CUDA-compatible GPU (e.g., NVIDIA A100, L4, or V100) with at least 16GB of VRAM.

```bash
# Install core dependencies
pip install -U torch transformers datasets peft trl bitsandbytes pandas matplotlib accelerate
