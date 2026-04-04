# DPO + QLoRA — Llama-3-8B-Instruct on UltraFeedback

This project fine-tunes **`meta-llama/Meta-Llama-3-8B-Instruct`** with **DPO (Direct Preference Optimization)** and **QLoRA** (4-bit quantization + LoRA) on the [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) (`train_prefs`) dataset. Training metrics are recorded via TensorBoard.

## 1. Project Structure

```text
DPO-baseline/
├── configs/
│   └── dpo_qlora_llama3_8b.yaml
├── data/
│   ├── train_dpo.jsonl        (fallback local data)
│   └── eval_dpo.jsonl
├── scripts/
│   └── train_llama3_instruct.sh
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   └── train_dpo.py
├── README.md
└── requirements.txt
```

## 2. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- A CUDA-enabled GPU and a compatible PyTorch build are required.
- For bf16 training, your GPU should support bf16. If not, set `train.bf16=false` and `train.fp16=true`.
- Llama 3 model access requires HuggingFace authentication (`huggingface-cli login`) and accepting the model license.

## 3. Dataset

The dataset [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) is loaded **directly from the HuggingFace Hub** — no local download needed. The `train_prefs` split is used for training and `test_prefs` for evaluation.

To use local JSONL files instead, set `data.use_hub_dataset: false` and provide `data.train_file` / `data.eval_file` paths. Each JSONL line must contain `prompt`, `chosen`, and `rejected` fields.

## 4. Run Training

```bash
bash scripts/train_llama3_instruct.sh
```

Or run directly with optional config overrides:

```bash
python -m src.train_dpo --config configs/dpo_qlora_llama3_8b.yaml \
  --override train.learning_rate=1e-4 \
  --override train.num_train_epochs=2
```

Adapter weights and TensorBoard logs are saved under `train.output_dir` (default: `outputs/llama3_8b_instruct_dpo_qlora`).

## 5. Monitor Training Metrics

TensorBoard logs are written to `<output_dir>/runs/`. Launch TensorBoard to inspect runs:

```bash
tensorboard --logdir outputs/
```

Metrics logged per step include:
- `train/loss`, `eval/loss`
- `train/rewards/chosen`, `train/rewards/rejected`
- `train/rewards/accuracies`, `train/rewards/margins`
- `train/logps/chosen`, `train/logps/rejected`
- `train/learning_rate`
- `perf/peak_gpu_memory_gb`, `perf/time_per_step_s`, `perf/approx_tokens_per_s`
- `objective/kl` (estimated KL divergence between policy and reference)
- `response_lengths/chosen`, `response_lengths/rejected`, `response_lengths/diff`

## 6. Config Reference

The config file `configs/dpo_qlora_llama3_8b.yaml` has four sections:

| Section | Key settings |
|---|---|
| `model` | `model_name_or_path`, `bnb_4bit_*` (quantization) |
| `data` | `use_hub_dataset`, `hub_dataset_name`, `hub_train_split`, `hub_eval_split` |
| `lora` | `r`, `lora_alpha`, `lora_dropout`, `target_modules` |
| `train` | `output_dir`, `learning_rate`, `beta`, `report_to`, etc. |

## 7. Common Issues

1. **Out-of-memory (OOM)** — reduce `train.max_length`, increase `train.gradient_accumulation_steps`, or reduce `train.per_device_train_batch_size`.
2. **Flash attention error** — set `model.use_flash_attention_2: false`.
3. **Dtype mismatch** — switch between `bf16`/`fp16` depending on GPU support.
4. **Model auth failure** — run `huggingface-cli login` and accept the model license.
5. **Hub dataset unavailable** — set `data.use_hub_dataset: false` and supply local JSONL files.
