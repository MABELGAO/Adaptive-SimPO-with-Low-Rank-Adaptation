from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_ADAPTER_PATH = "outputs/llama3_8b_instruct_dpo_qlora/checkpoint-421"
DEFAULT_HUB_DATASET_NAME = "tatsu-lab/alpaca_eval"
DEFAULT_HUB_DATASET_CONFIG = "alpaca_eval"
DEFAULT_HUB_SPLIT = "eval"
DEFAULT_PROMPT_COLUMN = "instruction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a DPO LoRA adapter.")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help="Path to the LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name or path. If omitted, try adapter config, then fall back to the default.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Optional JSONL file containing prompts. Each row must include a prompt field.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/inference/alpaca_eval_dpo_responses.jsonl",
        help="Where to write generated responses.",
    )
    parser.add_argument(
        "--hub-dataset-name",
        type=str,
        default=DEFAULT_HUB_DATASET_NAME,
        help="Dataset to load when --input-file is not provided.",
    )
    parser.add_argument(
        "--hub-dataset-config",
        type=str,
        default=DEFAULT_HUB_DATASET_CONFIG,
        help="Optional dataset config/subset name for the hub dataset.",
    )
    parser.add_argument(
        "--hub-split",
        type=str,
        default=DEFAULT_HUB_SPLIT,
        help="Split to load from the hub dataset.",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default=DEFAULT_PROMPT_COLUMN,
        help="Prompt column name for the hub dataset or local JSONL input.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Ignored when --do-sample is false.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p value for sampling.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. By default generation is deterministic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args()


def resolve_base_model(base_model: str | None, adapter_path: Path) -> str:
    if base_model:
        return base_model

    try:
        peft_config = PeftConfig.from_pretrained(str(adapter_path))
        if peft_config.base_model_name_or_path:
            return peft_config.base_model_name_or_path
    except Exception:
        pass

    return DEFAULT_BASE_MODEL


def load_prompt_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {input_path}")

        records: list[dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get(args.prompt_column) or row.get("prompt")
                if prompt is None:
                    raise ValueError(
                        f"Missing prompt field in {input_path} at line {idx + 1}. "
                        f"Expected '{args.prompt_column}' or 'prompt'."
                    )
                records.append(
                    {
                        "id": row.get("id", idx),
                        "prompt": str(prompt).strip(),
                        "source": str(input_path),
                    }
                )
        return records

    dataset_kwargs: dict[str, Any] = {"path": args.hub_dataset_name, "split": args.hub_split}
    if args.hub_dataset_config:
        dataset_kwargs["name"] = args.hub_dataset_config
    dataset = load_dataset(**dataset_kwargs)

    records = []
    for idx, row in enumerate(dataset):
        prompt = row.get(args.prompt_column) or row.get("prompt")
        if prompt is None:
            raise ValueError(
                f"Missing prompt column '{args.prompt_column}' in dataset "
                f"{args.hub_dataset_name}:{args.hub_split}."
            )
        records.append(
            {
                "id": row.get("id", idx),
                "prompt": str(prompt).strip(),
                "source": f"{args.hub_dataset_name}:{args.hub_dataset_config or 'default'}:{args.hub_split}",
            }
        )
    return records


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def prepare_batch(tokenizer, prompts: list[str], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rendered_prompts = [
        tokenizer.apply_chat_template(
            build_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    batch = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_lengths = batch["attention_mask"].sum(dim=1)
    batch = {k: v.to(device) for k, v in batch.items()}
    return batch, input_lengths


def generate_responses(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    base_model: str,
    adapter_path: Path,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    output_rows: list[dict[str, Any]] = []

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        prompts = [row["prompt"] for row in batch_records]
        model_inputs, input_lengths = prepare_batch(tokenizer, prompts, device)

        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generation_kwargs)

        for row, output_ids, input_len in zip(batch_records, generated, input_lengths.tolist()):
            new_tokens = output_ids[int(input_len) :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            output_rows.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "response": response,
                    "base_model": base_model,
                    "adapter_path": str(adapter_path),
                    "input_source": row["source"],
                    "generation_config": generation_kwargs,
                }
            )

    return output_rows


def save_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    base_model = resolve_base_model(args.base_model, adapter_path)
    prompt_records = load_prompt_records(args)

    print(f"Loaded {len(prompt_records)} prompts.")
    print(f"Base model: {base_model}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output file: {args.output_file}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=get_torch_dtype(),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    rows = generate_responses(
        model=model,
        tokenizer=tokenizer,
        records=prompt_records,
        args=args,
        base_model=base_model,
        adapter_path=adapter_path,
    )
    save_jsonl(rows, Path(args.output_file))

    print(f"Saved {len(rows)} responses to {args.output_file}")


if __name__ == "__main__":
    main()
