from __future__ import annotations

import argparse
import inspect
import os
import time
from typing import Any, Dict

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
from trl import DPOConfig, DPOTrainer

from src.config import ProjectConfig, load_project_config
from src.dataset import load_dpo_datasets


_DEFAULT_BETA = 0.1


# ---------------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------------

class PeakGPUMemoryCallback(TrainerCallback):
    """Log peak GPU memory usage (in GB) at every logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logs["perf/peak_gpu_memory_gb"] = round(peak_mem_gb, 4)


class TrainingTimeCallback(TrainerCallback):
    """Track total training time, per-step time, and throughput."""

    def __init__(self) -> None:
        self._train_start: float = 0.0
        self._step_start: float = 0.0
        self._pending_step_times: list[float] = []

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.perf_counter()
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.perf_counter() - self._step_start
        self._pending_step_times.append(elapsed)
        self._step_start = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self._pending_step_times:
            avg_step_time = sum(self._pending_step_times) / len(self._pending_step_times)
            logs["perf/time_per_step_s"] = round(avg_step_time, 4)
            logs["perf/total_elapsed_s"] = round(time.perf_counter() - self._train_start, 2)
            # Tokens per second: batch_size * max_length / step_time (approximate)
            if avg_step_time > 0 and hasattr(args, "per_device_train_batch_size") and hasattr(args, "max_length"):
                tokens_per_step = (
                    args.per_device_train_batch_size
                    * getattr(args, "gradient_accumulation_steps", 1)
                    * args.max_length
                )
                logs["perf/approx_tokens_per_s"] = round(tokens_per_step / avg_step_time, 1)
            self._pending_step_times.clear()

    def on_train_end(self, args, state, control, **kwargs):
        total = time.perf_counter() - self._train_start
        print(f"\n[TrainingTimeCallback] Total DPO training time: {total:.2f}s "
              f"({total / 60:.2f} min)")


# ---------------------------------------------------------------------------
# Extended DPOTrainer with response-length and KL metrics
# ---------------------------------------------------------------------------

class DPOTrainerWithExtendedMetrics(DPOTrainer):
    """Extends DPOTrainer to log response lengths and a KL divergence estimate."""

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        prefix = "eval_" if train_eval == "eval" else ""

        # ---- Response length metrics ----------------------------------------
        # TRL >= 0.9 uses per-side label tensors (chosen_labels, rejected_labels).
        # The labels tensor has -100 for prompt tokens and padding – only actual
        # response token positions have valid label ids.
        chosen_labels = batch.get("chosen_labels")
        rejected_labels = batch.get("rejected_labels")

        # Fallback: concatenated format used by older TRL versions
        if chosen_labels is None and "concatenated_labels" in batch:
            concat = batch["concatenated_labels"]
            bs = concat.size(0) // 2
            chosen_labels = concat[:bs]
            rejected_labels = concat[bs:]

        if chosen_labels is not None and rejected_labels is not None:
            chosen_len = (chosen_labels != -100).sum(dim=-1).float().mean()
            rejected_len = (rejected_labels != -100).sum(dim=-1).float().mean()
            metrics[f"{prefix}response_lengths/chosen"] = chosen_len.item()
            metrics[f"{prefix}response_lengths/rejected"] = rejected_len.item()
            metrics[f"{prefix}response_lengths/diff"] = (chosen_len - rejected_len).item()

        # ---- KL divergence estimate ------------------------------------------
        # rewards/chosen  = β * (log π_θ(y_w|x) − log π_ref(y_w|x))
        # rewards/rejected = β * (log π_θ(y_l|x) − log π_ref(y_l|x))
        # KL(π_θ ∥ π_ref) ≈ mean of log-ratios across observed responses.
        beta = getattr(self, "beta", None) or getattr(self.args, "beta", _DEFAULT_BETA)
        r_chosen = metrics.get(f"{prefix}rewards/chosen")
        r_rejected = metrics.get(f"{prefix}rewards/rejected")
        if r_chosen is not None and r_rejected is not None and beta != 0:
            kl_estimate = (r_chosen + r_rejected) / (2.0 * beta)
            metrics[f"{prefix}objective/kl"] = kl_estimate

        return loss, metrics


def _to_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype for 4-bit compute: {dtype_name}")


def _build_dpo_args(cfg: ProjectConfig) -> DPOConfig:
    train = cfg.train
    kwargs: Dict[str, Any] = {
        "output_dir": train.output_dir,
        "num_train_epochs": train.num_train_epochs,
        "per_device_train_batch_size": train.per_device_train_batch_size,
        "per_device_eval_batch_size": train.per_device_eval_batch_size,
        "gradient_accumulation_steps": train.gradient_accumulation_steps,
        "learning_rate": train.learning_rate,
        "lr_scheduler_type": train.lr_scheduler_type,
        "warmup_ratio": train.warmup_ratio,
        "weight_decay": train.weight_decay,
        "max_grad_norm": train.max_grad_norm,
        "logging_steps": train.logging_steps,
        "save_steps": train.save_steps,
        "eval_steps": train.eval_steps,
        "evaluation_strategy": train.evaluation_strategy,
        "save_strategy": train.save_strategy,
        "save_total_limit": train.save_total_limit,
        "seed": train.seed,
        "gradient_checkpointing": train.gradient_checkpointing,
        "bf16": train.bf16,
        "fp16": train.fp16,
        "max_length": train.max_length,
        "beta": train.beta,
        "report_to": train.report_to,
        "remove_unused_columns": train.remove_unused_columns,
    }

    signature = inspect.signature(DPOConfig.__init__).parameters
    if "tf32" in signature:
        kwargs["tf32"] = train.tf32

    return DPOConfig(**kwargs)


def _build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    dpo_args,
    peft_config,
    callbacks=None,
) -> DPOTrainerWithExtendedMetrics:
    kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "callbacks": callbacks or [],
    }

    init_signature = inspect.signature(DPOTrainer.__init__).parameters
    if "tokenizer" in init_signature:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in init_signature:
        kwargs["processing_class"] = tokenizer

    return DPOTrainerWithExtendedMetrics(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU DPO with QLoRA using TRL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo_qlora_llama3_8b.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. train.learning_rate=1e-4. Can be used multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_project_config(args.config, args.override)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(cfg.train.seed)

    if cfg.train.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_to_torch_dtype(cfg.model.bnb_4bit_compute_dtype),
    )

    model_kwargs: Dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": cfg.model.trust_remote_code,
    }

    if cfg.model.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    train_dataset, eval_dataset = load_dpo_datasets(
        use_hub_dataset=cfg.data.use_hub_dataset,
        hub_dataset_name=cfg.data.hub_dataset_name,
        hub_train_split=cfg.data.hub_train_split,
        hub_eval_split=cfg.data.hub_eval_split,
        train_file=cfg.data.train_file,
        eval_file=cfg.data.eval_file,
        prompt_col=cfg.data.prompt_column,
        chosen_col=cfg.data.chosen_column,
        rejected_col=cfg.data.rejected_column,
    )

    callbacks = [PeakGPUMemoryCallback(), TrainingTimeCallback()]

    dpo_args = _build_dpo_args(cfg)
    trainer = _build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dpo_args=dpo_args,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)


if __name__ == "__main__":
    main()
