from __future__ import annotations

import argparse
import inspect
import os
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
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
        print(f"\n[TrainingTimeCallback] Total SimPO training time: {total:.2f}s "
              f"({total / 60:.2f} min)")


# ---------------------------------------------------------------------------
# Adaptive Margin SimPO trainer
# ---------------------------------------------------------------------------

class AdaptiveMarginSimPOTrainer(DPOTrainer):
    """Reference-free SimPO trainer with sample-wise adaptive margins."""

    def __init__(
        self,
        *args,
        gamma_base: float,
        gamma_max: float,
        margin_slope: float,
        margin_offset: float,
        score_gap_max: float,
        fallback_chosen_score: float,
        fallback_rejected_score: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma_base = gamma_base
        self.gamma_max = gamma_max
        self.margin_slope = margin_slope
        self.margin_offset = margin_offset
        self.score_gap_max = max(score_gap_max, 1e-6)
        self.fallback_chosen_score = fallback_chosen_score
        self.fallback_rejected_score = fallback_rejected_score

    @staticmethod
    def tokenize_row(features, *args, **kwargs):
        tokenized = DPOTrainer.tokenize_row(features, *args, **kwargs)
        if "chosen_score" in features:
            tokenized["chosen_score"] = features["chosen_score"]
        if "rejected_score" in features:
            tokenized["rejected_score"] = features["rejected_score"]
        return tokenized

    @staticmethod
    def _extract_response_lengths(batch) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        chosen_labels = batch.get("chosen_labels")
        rejected_labels = batch.get("rejected_labels")

        if chosen_labels is None and "concatenated_labels" in batch:
            concat = batch["concatenated_labels"]
            bs = concat.size(0) // 2
            chosen_labels = concat[:bs]
            rejected_labels = concat[bs:]

        if chosen_labels is None or rejected_labels is None:
            return None, None

        chosen_len = (chosen_labels != -100).sum(dim=-1).clamp(min=1).float()
        rejected_len = (rejected_labels != -100).sum(dim=-1).clamp(min=1).float()
        return chosen_len, rejected_len

    @staticmethod
    def _get_batch_size(batch) -> int:
        for key in ("prompt_input_ids", "chosen_labels", "rejected_labels", "chosen_input_ids", "rejected_input_ids"):
            value = batch.get(key)
            if value is not None:
                return value.shape[0]
        raise ValueError("Unable to infer batch size from DPO batch.")

    @staticmethod
    def _score_to_tensor(values, batch_size: int, device: torch.device, fallback: float) -> torch.Tensor:
        if values is None:
            return torch.full((batch_size,), fallback, device=device, dtype=torch.float32)

        if torch.is_tensor(values):
            tensor = values.to(device=device, dtype=torch.float32)
            if tensor.ndim == 0:
                tensor = tensor.repeat(batch_size)
            return torch.nan_to_num(tensor, nan=fallback)

        processed = []
        for value in values:
            if value is None or (isinstance(value, str) and not value.strip()):
                processed.append(fallback)
            else:
                processed.append(float(value))
        return torch.tensor(processed, device=device, dtype=torch.float32)

    def _compute_adaptive_gamma(
        self,
        chosen_scores: torch.Tensor,
        rejected_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_gap = torch.clamp(chosen_scores - rejected_scores, min=0.0)
        normalized_gap = torch.clamp(raw_gap / self.score_gap_max, min=0.0, max=1.0)
        gamma = self.gamma_max * torch.sigmoid(self.margin_slope * (normalized_gap - self.margin_offset))
        gamma = gamma + self.gamma_base
        return gamma, normalized_gap, raw_gap

    def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
        prefix = "eval_" if train_eval == "eval" else ""
        forward_output = self.concatenated_forward(model, batch)
        if not isinstance(forward_output, tuple) or len(forward_output) < 4:
            raise TypeError("Unexpected DPOTrainer.concatenated_forward output.")

        policy_chosen_logps = forward_output[0]
        policy_rejected_logps = forward_output[1]
        policy_chosen_logits = forward_output[2]
        policy_rejected_logits = forward_output[3]

        chosen_len, rejected_len = self._extract_response_lengths(batch)
        if chosen_len is None or rejected_len is None:
            raise ValueError("Adaptive Margin SimPO requires chosen/rejected label tensors to compute length-normalized rewards.")

        beta = getattr(self, "beta", None) or getattr(self.args, "beta", _DEFAULT_BETA)
        chosen_avg_logps = policy_chosen_logps / chosen_len
        rejected_avg_logps = policy_rejected_logps / rejected_len
        chosen_rewards = beta * chosen_avg_logps
        rejected_rewards = beta * rejected_avg_logps

        batch_size = self._get_batch_size(batch)
        device = policy_chosen_logps.device
        chosen_scores = self._score_to_tensor(
            batch.get("chosen_score"),
            batch_size=batch_size,
            device=device,
            fallback=self.fallback_chosen_score,
        )
        rejected_scores = self._score_to_tensor(
            batch.get("rejected_score"),
            batch_size=batch_size,
            device=device,
            fallback=self.fallback_rejected_score,
        )
        adaptive_gamma, normalized_gap, raw_gap = self._compute_adaptive_gamma(chosen_scores, rejected_scores)

        reward_margins = chosen_rewards - rejected_rewards
        losses = -F.logsigmoid(reward_margins - adaptive_gamma)
        reward_accuracies = (reward_margins > 0).float()

        metrics = {
            f"{prefix}rewards/chosen": chosen_rewards.detach().mean().item(),
            f"{prefix}rewards/rejected": rejected_rewards.detach().mean().item(),
            f"{prefix}rewards/accuracies": reward_accuracies.detach().mean().item(),
            f"{prefix}rewards/margins": reward_margins.detach().mean().item(),
            f"{prefix}logps/chosen": policy_chosen_logps.detach().mean().item(),
            f"{prefix}logps/rejected": policy_rejected_logps.detach().mean().item(),
            f"{prefix}avg_logps/chosen": chosen_avg_logps.detach().mean().item(),
            f"{prefix}avg_logps/rejected": rejected_avg_logps.detach().mean().item(),
            f"{prefix}logits/chosen": policy_chosen_logits.detach().mean().item(),
            f"{prefix}logits/rejected": policy_rejected_logits.detach().mean().item(),
            f"{prefix}response_lengths/chosen": chosen_len.detach().mean().item(),
            f"{prefix}response_lengths/rejected": rejected_len.detach().mean().item(),
            f"{prefix}response_lengths/diff": (chosen_len - rejected_len).detach().mean().item(),
            f"{prefix}objective/gamma": adaptive_gamma.detach().mean().item(),
            f"{prefix}objective/score_gap": normalized_gap.detach().mean().item(),
            f"{prefix}objective/raw_score_gap": raw_gap.detach().mean().item(),
        }

        return losses.mean(), metrics


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
    cfg,
    score_gap_max,
    callbacks=None,
) -> AdaptiveMarginSimPOTrainer:
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

    return AdaptiveMarginSimPOTrainer(
        gamma_base=cfg.simpo.gamma_base,
        gamma_max=cfg.simpo.gamma_max,
        margin_slope=cfg.simpo.margin_slope,
        margin_offset=cfg.simpo.margin_offset,
        score_gap_max=score_gap_max,
        fallback_chosen_score=cfg.simpo.fallback_chosen_score,
        fallback_rejected_score=cfg.simpo.fallback_rejected_score,
        **kwargs,
    )


def _compute_score_gap_max(train_dataset, fallback_chosen_score: float, fallback_rejected_score: float) -> float:
    max_gap = max(fallback_chosen_score - fallback_rejected_score, 0.0)
    chosen_scores = train_dataset["chosen_score"] if "chosen_score" in train_dataset.column_names else []
    rejected_scores = train_dataset["rejected_score"] if "rejected_score" in train_dataset.column_names else []

    for chosen, rejected in zip(chosen_scores, rejected_scores):
        chosen_value = fallback_chosen_score if chosen is None else float(chosen)
        rejected_value = fallback_rejected_score if rejected is None else float(rejected)
        max_gap = max(max_gap, chosen_value - rejected_value)

    return max(max_gap, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU Adaptive Margin SimPO with QLoRA using TRL")
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
        chosen_score_col=cfg.data.chosen_score_column,
        rejected_score_col=cfg.data.rejected_score_column,
    )
    score_gap_max = _compute_score_gap_max(
        train_dataset,
        fallback_chosen_score=cfg.simpo.fallback_chosen_score,
        fallback_rejected_score=cfg.simpo.fallback_rejected_score,
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
        cfg=cfg,
        score_gap_max=score_gap_max,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)


if __name__ == "__main__":
    main()
