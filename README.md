# DPO + QLoRA — Llama-3-8B-Instruct

Fine-tunes **`meta-llama/Meta-Llama-3-8B-Instruct`** with **DPO** and **QLoRA** (4-bit + LoRA) on [`argilla/dpo-mix-7k`](https://huggingface.co/datasets/argilla/dpo-mix-7k). Training metrics are logged via TensorBoard.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> Requires a CUDA GPU. For non-bf16 GPUs set `train.bf16=false` and `train.fp16=true`. Run `huggingface-cli login` to access Llama 3.

## Dataset

Uses `train` / `test` splits, loaded directly from the Hub. To use local JSONL files, set `data.use_hub_dataset: false` and point `data.train_file` / `data.eval_file` to files with `prompt`, `chosen`, `rejected` fields.

## Training

```bash
bash scripts/train_llama3_instruct.sh
# or with overrides:
python -m src.train_dpo --config configs/dpo_qlora_llama3_8b.yaml \
  --override train.learning_rate=1e-4
```

Adapter weights are saved to `train.output_dir` (default: `outputs/llama3_8b_instruct_dpo_qlora`).

## Monitoring

```bash
tensorboard --logdir outputs/
```

Key metrics: `train/loss`, `eval/loss`, `train/rewards/*`, `train/logps/*`, `objective/kl`, `perf/*`.

## Config (`configs/dpo_qlora_llama3_8b.yaml`)

| Section | Key fields |
|---|---|
| `model` | `model_name_or_path`, `bnb_4bit_*` |
| `data` | `use_hub_dataset`, `hub_dataset_name`, `hub_train_split`, `hub_eval_split` |
| `lora` | `r`, `lora_alpha`, `lora_dropout`, `target_modules` |
| `train` | `output_dir`, `learning_rate`, `beta`, `bf16`, `max_length` |

## Common Issues

| Issue | Fix |
|---|---|
| OOM | Reduce `train.max_length` or `per_device_train_batch_size`; increase `gradient_accumulation_steps` |
| Flash attention error | Set `model.use_flash_attention_2: false` |
| Dtype mismatch | Switch `bf16`/`fp16` based on GPU support |
| Model auth failure | Run `huggingface-cli login` and accept the license |
| Hub dataset unavailable | Set `data.use_hub_dataset: false` and use local JSONL files |
