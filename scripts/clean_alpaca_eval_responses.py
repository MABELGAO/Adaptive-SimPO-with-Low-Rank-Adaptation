from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ASSISTANT_SPLIT_RE = re.compile(r"assistant\s*\n+", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean AlpacaEval-style response JSONL files with chat-template leakage."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the original JSONL response file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to write the cleaned JSONL file.",
    )
    parser.add_argument(
        "--drop-raw",
        action="store_true",
        help="Do not keep the original response in a response_raw field.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Keep only id, prompt, and cleaned response in the output rows.",
    )
    return parser.parse_args()


def clean_response(prompt: str, response: str) -> tuple[str, str]:
    original = response.strip()
    cleaned = original
    rule = "unchanged"

    assistant_matches = list(ASSISTANT_SPLIT_RE.finditer(cleaned))
    if assistant_matches:
        cleaned = cleaned[assistant_matches[-1].end() :].strip()
        rule = "after_last_assistant_tag"

    if cleaned.lower().startswith("assistant"):
        cleaned = cleaned[len("assistant") :].lstrip(" \n:：-").strip()
        rule = "leading_assistant_tag"

    if cleaned == original and prompt:
        prompt_idx = cleaned.find(prompt)
        if prompt_idx != -1:
            cleaned = cleaned[prompt_idx + len(prompt) :].lstrip(" \n:：-").strip()
            rule = "after_prompt_echo"

    return cleaned, rule


def clean_rows(
    rows: list[dict[str, Any]],
    *,
    drop_raw: bool = False,
    minimal: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "rows_total": 0,
        "rows_changed": 0,
        "rows_unchanged": 0,
        "empty_after_cleaning": 0,
    }

    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response", ""))
        cleaned_response, rule = clean_response(prompt, response)

        if minimal:
            cleaned_row = {
                "id": row.get("id"),
                "prompt": row.get("prompt"),
                "response": cleaned_response,
            }
        else:
            cleaned_row = dict(row)
            cleaned_row["response"] = cleaned_response
            cleaned_row["cleaning_rule"] = rule
            if not drop_raw:
                cleaned_row["response_raw"] = response

        stats["rows_total"] += 1
        if cleaned_response != response.strip():
            stats["rows_changed"] += 1
        else:
            stats["rows_unchanged"] += 1
        if not cleaned_response:
            stats["empty_after_cleaning"] += 1

        cleaned_rows.append(cleaned_row)

    return cleaned_rows, stats


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    rows = load_jsonl(input_path)
    cleaned_rows, stats = clean_rows(
        rows,
        drop_raw=args.drop_raw,
        minimal=args.minimal,
    )
    save_jsonl(output_path, cleaned_rows)

    print(f"Loaded {stats['rows_total']} rows from {input_path}")
    print(f"Changed {stats['rows_changed']} rows")
    print(f"Unchanged {stats['rows_unchanged']} rows")
    print(f"Empty after cleaning: {stats['empty_after_cleaning']}")
    print(f"Saved cleaned rows to {output_path}")


if __name__ == "__main__":
    main()
