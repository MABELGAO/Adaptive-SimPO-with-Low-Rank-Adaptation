from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from peft import PeftConfig
from transformers import AutoTokenizer, set_seed


DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_ADAPTER_PATH = "outputs/llama3_8b_instruct_dpo_qlora/checkpoint-421"
DEFAULT_HUB_DATASET_NAME = "tatsu-lab/alpaca_eval"
DEFAULT_HUB_DATASET_CONFIG = "alpaca_eval"
DEFAULT_HUB_SPLIT = "eval"
DEFAULT_PROMPT_COLUMN = "instruction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched vLLM inference with a LoRA adapter.")
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
        default="outputs/inference/alpaca_eval_vllm_responses.jsonl",
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
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Target GPU memory utilization for vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model length for vLLM.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
        help="Maximum number of concurrent sequences handled by vLLM.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Inference dtype passed to vLLM.",
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


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def build_prompts(tokenizer, records: list[dict[str, Any]]) -> list[str]:
    tokenizer.padding_side = "left"
    return [
        tokenizer.apply_chat_template(
            build_messages(record["prompt"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for record in records
    ]


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

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed. Install it first, then rerun this script."
        ) from exc

    base_model = resolve_base_model(args.base_model, adapter_path)
    prompt_records = load_prompt_records(args)

    print(f"Loaded {len(prompt_records)} prompts.")
    print(f"Base model: {base_model}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output file: {args.output_file}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rendered_prompts = build_prompts(tokenizer, prompt_records)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 1.0,
    )

    llm = LLM(
        model=base_model,
        tokenizer=base_model,
        enable_lora=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        trust_remote_code=False,
    )

    lora_request = LoRARequest("adapter", 1, str(adapter_path))
    outputs = llm.generate(rendered_prompts, sampling_params, lora_request=lora_request)

    rows: list[dict[str, Any]] = []
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else 0.0,
        "top_p": args.top_p if args.do_sample else 1.0,
        "engine": "vllm",
    }

    for record, output in zip(prompt_records, outputs):
        response = output.outputs[0].text.strip() if output.outputs else ""
        rows.append(
            {
                "id": record["id"],
                "prompt": record["prompt"],
                "response": response,
                "base_model": base_model,
                "adapter_path": str(adapter_path),
                "input_source": record["source"],
                "generation_config": generation_config,
            }
        )

    save_jsonl(rows, Path(args.output_file))
    print(f"Saved {len(rows)} responses to {args.output_file}")


if __name__ == "__main__":
    main()
