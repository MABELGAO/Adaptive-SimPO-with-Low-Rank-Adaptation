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
```

## Hugging Face Authentication
Since Llama-3 is a gated model, you must authenticate your environment:

Accept the EULA on the Meta-Llama-3-8B-Instruct Hugging Face page.

Generate a Read Access Token in your HF settings.

Login via terminal:

``` Bash
huggingface-cli login
```

## Usage
Run the script directly. It will automatically handle directory creation, dataset mapping, tuning, and formal training.

```Bash
python run_simpo_llama3.py
```

## Potential Issues & Troubleshooting
1. GatedRepoError: 401/403 Client Error
Cause: The script cannot access the Llama-3 repository because the environment is unauthorized or the Meta EULA has not been signed.

Solution: Ensure you have clicked "Agree and access repository" on the Llama-3 HF page. Run huggingface-cli login in your terminal and paste your Access Token. Alternatively, inject login("your_hf_token") at the top of the script.

2. ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes
Cause: Running the script on a machine without a CUDA-enabled NVIDIA GPU (e.g., a local Mac or standard CPU server).

Solution: bitsandbytes strictly requires CUDA. For local Dry-Runs on macOS/CPU, remove BitsAndBytesConfig and load a smaller model (e.g., Qwen/Qwen2.5-0.5B-Instruct) in standard precision. Use cloud instances (AWS, GCP, RunPod) for the formal training.

3. TRLExperimentalWarning on Startup
Cause: The script imports CPOTrainer from trl.experimental.cpo.

Solution: This is a standard warning indicating the API is actively being developed by Hugging Face. It does not affect your training. To silence it, set the environment variable: export TRL_EXPERIMENTAL_SILENCE=1.

4. CUDA Out of Memory (OOM)
Cause: The GPU lacks sufficient VRAM for an 8B model even with 4-bit quantization, or the micro-batch size is too large.

Solution: Decrease BATCH_SIZE to 1. To maintain the same training dynamics and global batch size, proportionally increase GRAD_ACCUM_STEPS (e.g., if Batch Size goes from 2 to 1, double the accumulation steps).

5. Blank Matplotlib Charts
Cause: If total training steps are lower than eval_steps, the Trainer will not log enough data points to draw a continuous line.

Solution: This v2 script mitigates this by lowering eval_steps=5 and adding marker='o'. If you run a tiny subset of data for testing, ensure eval_steps is smaller than your total steps.
