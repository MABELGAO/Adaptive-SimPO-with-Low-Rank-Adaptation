from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute reference chosen/rejected sequence log-probs for local DPO/SimPO JSONL datasets."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Path to the source JSONL file.")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to write the augmented JSONL file.")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Reference model used to compute log-probs.")
    parser.add_argument("--prompt-column", type=str, default="prompt", help="Prompt column name.")
    parser.add_argument("--chosen-column", type=str, default="chosen", help="Chosen response column name.")
    parser.add_argument("--rejected-column", type=str, default="rejected", help="Rejected response column name.")
    parser.add_argument("--ref-chosen-logp-column", type=str, default="ref_chosen_logp", help="Output column for chosen log-probs.")
    parser.add_argument("--ref-rejected-logp-column", type=str, default="ref_rejected_logp", help="Output column for rejected log-probs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for reference forward passes.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length for prompt+response.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to model/tokenizer loading.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the reference model in 4-bit to reduce GPU memory.")
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Compute dtype when --load-in-4bit is enabled.",
    )
    return parser.parse_args()


def _to_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Each JSONL row must be an object. Found {type(row).__name__} on line {line_no}.")
            rows.append(row)
    return rows


def _validate_rows(rows: list[dict[str, Any]], prompt_column: str, chosen_column: str, rejected_column: str) -> None:
    required = (prompt_column, chosen_column, rejected_column)
    for idx, row in enumerate(rows, start=1):
        missing = [column for column in required if column not in row]
        if missing:
            raise ValueError(f"Row {idx} is missing required columns {missing}.")


def _build_feature(
    tokenizer,
    prompt: str,
    response: str,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(str(prompt), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(str(response), add_special_tokens=False)["input_ids"]

    input_ids = list(prompt_ids) + list(response_ids)
    labels = [-100] * len(prompt_ids) + list(response_ids)

    if tokenizer.bos_token_id is not None:
        input_ids = [tokenizer.bos_token_id] + input_ids
        labels = [-100] + labels
    if tokenizer.eos_token_id is not None:
        input_ids = input_ids + [tokenizer.eos_token_id]
        labels = labels + [tokenizer.eos_token_id]

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _collate(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids = []
    attention_mask = []
    labels = []
    for feature in features:
        pad = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(feature["attention_mask"] + [0] * pad)
        labels.append(feature["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _sequence_logps(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:].clone()
    valid_mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_logps = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * valid_mask).sum(dim=-1)


def _load_reference_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_to_torch_dtype(args.bnb_4bit_compute_dtype),
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.eval()
    return tokenizer, model


def _model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Reference model has no parameters to infer an input device from.") from exc


def precompute_ref_logps(args: argparse.Namespace) -> None:
    rows = _read_jsonl(args.input_file)
    _validate_rows(rows, args.prompt_column, args.chosen_column, args.rejected_column)
    tokenizer, model = _load_reference_model(args)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    total = len(rows)
    with torch.inference_mode():
        for start in range(0, total, args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            chosen_features = [
                _build_feature(
                    tokenizer,
                    prompt=str(row[args.prompt_column]),
                    response=str(row[args.chosen_column]),
                    max_length=args.max_length,
                )
                for row in batch_rows
            ]
            rejected_features = [
                _build_feature(
                    tokenizer,
                    prompt=str(row[args.prompt_column]),
                    response=str(row[args.rejected_column]),
                    max_length=args.max_length,
                )
                for row in batch_rows
            ]
            merged_features = chosen_features + rejected_features
            batch = _collate(merged_features, pad_token_id=pad_token_id)
            device = _model_input_device(model)
            batch = {key: value.to(device) for key, value in batch.items()}
            seq_logps = _sequence_logps(model, batch).detach().cpu().tolist()
            midpoint = len(batch_rows)
            chosen_logps = seq_logps[:midpoint]
            rejected_logps = seq_logps[midpoint:]

            for row, chosen_logp, rejected_logp in zip(batch_rows, chosen_logps, rejected_logps):
                row[args.ref_chosen_logp_column] = float(chosen_logp)
                row[args.ref_rejected_logp_column] = float(rejected_logp)

            end = min(start + args.batch_size, total)
            print(f"Processed {end}/{total} rows")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote reference log-probs to {args.output_file}")


def main() -> None:
    args = parse_args()
    precompute_ref_logps(args)


if __name__ == "__main__":
    main()
