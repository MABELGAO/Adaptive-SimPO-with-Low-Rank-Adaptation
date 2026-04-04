from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    trust_remote_code: bool = False
    use_flash_attention_2: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class DataConfig:
    # Hub dataset settings (takes priority when use_hub_dataset=True)
    use_hub_dataset: bool = True
    hub_dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    hub_train_split: str = "train_prefs"
    hub_eval_split: str = "test_prefs"
    # Local file settings (used when use_hub_dataset=False)
    train_file: str = "data/train_dpo.jsonl"
    eval_file: str = "data/eval_dpo.jsonl"
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    shuffle_train: bool = True
    train_sample_seed: int = 42


@dataclass
class LoraArguments:
    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainConfig:
    output_dir: str = "outputs/llama3_8b_dpo_qlora"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    save_steps: int = 100
    eval_steps: int = 100
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 2
    seed: int = 42
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    max_length: int = 2048
    max_prompt_length: int = 1024
    max_target_length: int = 1024
    beta: float = 0.1
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    remove_unused_columns: bool = False
    resume_from_checkpoint: Optional[str] = None


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraArguments = field(default_factory=LoraArguments)
    train: TrainConfig = field(default_factory=TrainConfig)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(v.strip()) for v in inner.split(",")]

    return value


def _apply_override(config: Dict[str, Any], expr: str) -> None:
    if "=" not in expr:
        raise ValueError(f"Invalid override: {expr}. Expected key=value format.")

    key, raw_value = expr.split("=", 1)
    keys = key.split(".")
    value = _parse_scalar(raw_value)

    target = config
    for part in keys[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[keys[-1]] = value


def load_project_config(config_path: str, overrides: Optional[List[str]] = None) -> ProjectConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    for item in overrides or []:
        _apply_override(raw, item)

    model = ModelConfig(**raw.get("model", {}))
    data = DataConfig(**raw.get("data", {}))
    lora = LoraArguments(**raw.get("lora", {}))
    train = TrainConfig(**raw.get("train", {}))
    return ProjectConfig(model=model, data=data, lora=lora, train=train)
