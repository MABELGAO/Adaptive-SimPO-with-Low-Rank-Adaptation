from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_JUDGE_MODEL = "qwen-max"
DEFAULT_OUTPUT_DIR = "outputs/judgments"
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_RETRIES = 3


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator comparing two assistant responses to the same user prompt.

Evaluate which response is better overall. Prioritize:
1. Correctness and factual accuracy
2. Relevance to the user's request
3. Helpfulness and completeness
4. Clarity and coherence
5. Safety

Do not prefer a response just because it is longer, more detailed, or more verbose.
A concise response can be better if it fully and accurately answers the user's request.
Penalize hallucinations, unsupported claims, irrelevance, evasiveness, and unsafe advice.
If both responses are similarly good or similarly flawed, return Tie. Do not force a winner.

You must respond with valid JSON only, following this schema:
{"winner":"A|B|Tie","reason":"short explanation"}
"""


@dataclass(frozen=True)
class ResponseRecord:
    id: Any
    prompt: str
    response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise LLM-as-a-Judge evaluation for response JSONL files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_eval_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help="Judge model name.")
        subparser.add_argument(
            "--api-base",
            type=str,
            default=DEFAULT_API_BASE,
            help="OpenAI-compatible API base URL.",
        )
        subparser.add_argument(
            "--api-key-env",
            type=str,
            default=DEFAULT_API_KEY_ENV,
            help="Environment variable that stores the API key.",
        )
        subparser.add_argument(
            "--output-dir",
            type=str,
            default=DEFAULT_OUTPUT_DIR,
            help="Directory for judgment outputs.",
        )
        subparser.add_argument("--seed", type=int, default=42, help="Random seed for display-order randomization.")
        subparser.add_argument(
            "--max-samples",
            type=int,
            default=None,
            help="Optional cap on the number of aligned samples to judge.",
        )
        subparser.add_argument(
            "--temperature",
            type=float,
            default=0.0,
            help="Judge generation temperature. Use 0.0 for maximum determinism.",
        )
        subparser.add_argument(
            "--timeout",
            type=int,
            default=DEFAULT_TIMEOUT_S,
            help="API timeout in seconds.",
        )
        subparser.add_argument(
            "--max-retries",
            type=int,
            default=DEFAULT_MAX_RETRIES,
            help="Maximum retries per API call.",
        )
        subparser.add_argument(
            "--resume",
            action="store_true",
            help="Skip ids already present in an existing judgments file.",
        )

    pair_parser = subparsers.add_parser("pair", help="Run one pairwise comparison.")
    pair_parser.add_argument("--file-a", type=str, required=True, help="JSONL file for model A.")
    pair_parser.add_argument("--file-b", type=str, required=True, help="JSONL file for model B.")
    pair_parser.add_argument("--name-a", type=str, required=True, help="Display name for model A.")
    pair_parser.add_argument("--name-b", type=str, required=True, help="Display name for model B.")
    add_common_eval_args(pair_parser)

    trio_parser = subparsers.add_parser("trio", help="Run all three pairwise comparisons among three models.")
    trio_parser.add_argument("--file-a", type=str, required=True, help="JSONL file for model A.")
    trio_parser.add_argument("--file-b", type=str, required=True, help="JSONL file for model B.")
    trio_parser.add_argument("--file-c", type=str, required=True, help="JSONL file for model C.")
    trio_parser.add_argument("--name-a", type=str, required=True, help="Display name for model A.")
    trio_parser.add_argument("--name-b", type=str, required=True, help="Display name for model B.")
    trio_parser.add_argument("--name-c", type=str, required=True, help="Display name for model C.")
    add_common_eval_args(trio_parser)

    return parser.parse_args()


def slugify(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in value.strip()]
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "model"


def load_response_file(path: Path) -> dict[Any, ResponseRecord]:
    records: dict[Any, ResponseRecord] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            record_id = row.get("id")
            prompt = row.get("prompt")
            response = row.get("response")
            if record_id is None or prompt is None or response is None:
                raise ValueError(
                    f"{path} line {line_no} must contain only-necessary fields id/prompt/response at minimum."
                )
            if record_id in records:
                raise ValueError(f"Duplicate id {record_id!r} found in {path}.")
            records[record_id] = ResponseRecord(
                id=record_id,
                prompt=str(prompt),
                response=str(response),
            )
    return records


def align_records(
    records_a: dict[Any, ResponseRecord],
    records_b: dict[Any, ResponseRecord],
) -> tuple[list[tuple[ResponseRecord, ResponseRecord]], dict[str, Any]]:
    ids_a = set(records_a)
    ids_b = set(records_b)
    common_ids = sorted(ids_a & ids_b, key=lambda x: str(x))
    only_a = sorted(ids_a - ids_b, key=lambda x: str(x))
    only_b = sorted(ids_b - ids_a, key=lambda x: str(x))

    prompt_mismatches: list[Any] = []
    aligned: list[tuple[ResponseRecord, ResponseRecord]] = []
    for record_id in common_ids:
        rec_a = records_a[record_id]
        rec_b = records_b[record_id]
        if rec_a.prompt != rec_b.prompt:
            prompt_mismatches.append(record_id)
            continue
        aligned.append((rec_a, rec_b))

    meta = {
        "rows_a": len(records_a),
        "rows_b": len(records_b),
        "common_ids": len(common_ids),
        "aligned_ids": len(aligned),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "prompt_mismatch_ids": prompt_mismatches,
    }
    return aligned, meta


def response_length_metrics(text: str) -> dict[str, int]:
    stripped = text.strip()
    return {
        "chars": len(stripped),
        "words": len(stripped.split()) if stripped else 0,
        "lines": len(stripped.splitlines()) if stripped else 0,
    }


def build_user_prompt(prompt: str, response_a: str, response_b: str) -> str:
    return (
        "Compare the two responses to the same user prompt.\n\n"
        f"[User Prompt]\n{prompt}\n\n"
        f"[Response A]\n{response_a}\n\n"
        f"[Response B]\n{response_b}\n\n"
        'Return JSON only in the form {"winner":"A|B|Tie","reason":"short explanation"}.'
    )


def call_judge_api(
    *,
    api_base: str,
    api_key: str,
    judge_model: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, Any]]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return str(content), data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Judge API call failed after {max_retries} attempts: {last_error}") from last_error


def parse_judge_output(raw_text: str) -> tuple[str, str]:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        winner = str(parsed.get("winner", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
    except json.JSONDecodeError:
        lowered = text.lower()
        if '"winner":"a"' in lowered or '"winner": "a"' in lowered:
            winner = "A"
        elif '"winner":"b"' in lowered or '"winner": "b"' in lowered:
            winner = "B"
        elif '"winner":"tie"' in lowered or '"winner": "tie"' in lowered:
            winner = "Tie"
        elif "winner: a" in lowered or "response a is better" in lowered or "i choose a" in lowered:
            winner = "A"
        elif "winner: b" in lowered or "response b is better" in lowered or "i choose b" in lowered:
            winner = "B"
        elif (
            "tie" in lowered
            or "equally good" in lowered
            or "equally strong" in lowered
            or "similarly good" in lowered
            or "similar quality" in lowered
            or "both responses are about equally good" in lowered
        ):
            winner = "Tie"
        else:
            winner = "Invalid"
        reason = text

    normalized = winner.upper()
    if normalized == "A":
        return "A", reason
    if normalized == "B":
        return "B", reason
    if normalized == "TIE":
        return "Tie", reason
    return "Invalid", reason


def load_existing_judgments(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed.add(str(row["id"]))
    return completed


def save_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def summarize_judgments(rows: list[dict[str, Any]], *, name_a: str, name_b: str) -> dict[str, Any]:
    wins_a = sum(1 for row in rows if row["result"] == "A_win")
    wins_b = sum(1 for row in rows if row["result"] == "B_win")
    ties = sum(1 for row in rows if row["result"] == "Tie")
    invalid = sum(1 for row in rows if row["result"] == "Invalid")
    total = len(rows)
    decided = wins_a + wins_b
    adjusted_total = wins_a + wins_b + ties

    return {
        "model_a": name_a,
        "model_b": name_b,
        "total_samples": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "invalid": invalid,
        "win_rate_a_excluding_ties": (wins_a / decided) if decided else None,
        "win_rate_b_excluding_ties": (wins_b / decided) if decided else None,
        "adjusted_win_rate_a": ((wins_a + 0.5 * ties) / adjusted_total) if adjusted_total else None,
        "adjusted_win_rate_b": ((wins_b + 0.5 * ties) / adjusted_total) if adjusted_total else None,
    }


def should_display_model_a_first(*, seed: int, comparison_slug: str, sample_id: Any) -> bool:
    key = f"{seed}|{comparison_slug}|{sample_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return digest[0] % 2 == 0


def judge_one_pair(
    *,
    rec_a: ResponseRecord,
    rec_b: ResponseRecord,
    name_a: str,
    name_b: str,
    comparison_slug: str,
    seed: int,
    api_base: str,
    api_key: str,
    judge_model: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    display_a_is_model_a = should_display_model_a_first(
        seed=seed,
        comparison_slug=comparison_slug,
        sample_id=rec_a.id,
    )
    shown_a = rec_a if display_a_is_model_a else rec_b
    shown_b = rec_b if display_a_is_model_a else rec_a

    raw_text, _ = call_judge_api(
        api_base=api_base,
        api_key=api_key,
        judge_model=judge_model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=build_user_prompt(rec_a.prompt, shown_a.response, shown_b.response),
    )
    judge_winner, judge_reason = parse_judge_output(raw_text)

    if judge_winner == "Tie":
        result = "Tie"
        winner_model = "Tie"
    elif judge_winner == "Invalid":
        result = "Invalid"
        winner_model = "Invalid"
    else:
        winner_is_model_a = (judge_winner == "A" and display_a_is_model_a) or (
            judge_winner == "B" and not display_a_is_model_a
        )
        result = "A_win" if winner_is_model_a else "B_win"
        winner_model = name_a if winner_is_model_a else name_b

    len_a = response_length_metrics(rec_a.response)
    len_b = response_length_metrics(rec_b.response)

    return {
        "id": rec_a.id,
        "prompt": rec_a.prompt,
        "model_a": name_a,
        "model_b": name_b,
        "response_a": rec_a.response,
        "response_b": rec_b.response,
        "response_a_chars": len_a["chars"],
        "response_b_chars": len_b["chars"],
        "response_a_words": len_a["words"],
        "response_b_words": len_b["words"],
        "length_diff_chars": len_a["chars"] - len_b["chars"],
        "length_diff_words": len_a["words"] - len_b["words"],
        "display_order": {
            "A": name_a if display_a_is_model_a else name_b,
            "B": name_b if display_a_is_model_a else name_a,
        },
        "judge_model": judge_model,
        "judge_winner_label": judge_winner,
        "winner_model": winner_model,
        "result": result,
        "judge_reason": judge_reason,
        "judge_raw": raw_text,
    }


def run_pairwise_comparison(
    *,
    file_a: Path,
    file_b: Path,
    name_a: str,
    name_b: str,
    judge_model: str,
    api_base: str,
    api_key: str,
    output_dir: Path,
    seed: int,
    max_samples: int | None,
    temperature: float,
    timeout: int,
    max_retries: int,
    resume: bool,
) -> dict[str, Any]:
    records_a = load_response_file(file_a)
    records_b = load_response_file(file_b)
    aligned_pairs, alignment_meta = align_records(records_a, records_b)

    if max_samples is not None:
        aligned_pairs = aligned_pairs[:max_samples]

    comparison_slug = f"{slugify(name_a)}_vs_{slugify(name_b)}"
    judgments_path = output_dir / f"{comparison_slug}.judgments.jsonl"
    summary_path = output_dir / f"{comparison_slug}.summary.json"

    completed_ids = load_existing_judgments(judgments_path) if resume else set()
    all_rows: list[dict[str, Any]] = []
    if resume and judgments_path.exists():
        with judgments_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))

    new_rows: list[dict[str, Any]] = []
    for rec_a, rec_b in aligned_pairs:
        if str(rec_a.id) in completed_ids:
            continue
        row = judge_one_pair(
            rec_a=rec_a,
            rec_b=rec_b,
            name_a=name_a,
            name_b=name_b,
            comparison_slug=comparison_slug,
            seed=seed,
            api_base=api_base,
            api_key=api_key,
            judge_model=judge_model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        new_rows.append(row)

    if new_rows:
        save_jsonl(judgments_path, new_rows, append=resume and judgments_path.exists())
        all_rows.extend(new_rows)
    elif not resume:
        save_jsonl(judgments_path, [], append=False)

    summary = summarize_judgments(all_rows, name_a=name_a, name_b=name_b)
    summary.update(
        {
            "comparison": comparison_slug,
            "file_a": str(file_a),
            "file_b": str(file_b),
            "judge_model": judge_model,
            "api_base": api_base,
            "seed": seed,
            "max_samples": max_samples,
            "alignment": alignment_meta,
            "judgments_file": str(judgments_path),
        }
    )
    save_json(summary_path, summary)
    print(f"[{comparison_slug}] saved {len(all_rows)} judgments")
    print(
        f"[{comparison_slug}] wins_a={summary['wins_a']} wins_b={summary['wins_b']} "
        f"ties={summary['ties']} invalid={summary['invalid']}"
    )
    return summary


def run_pair_mode(args: argparse.Namespace) -> None:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key in environment variable {args.api_key_env}.")

    run_pairwise_comparison(
        file_a=Path(args.file_a),
        file_b=Path(args.file_b),
        name_a=args.name_a,
        name_b=args.name_b,
        judge_model=args.judge_model,
        api_base=args.api_base,
        api_key=api_key,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        max_samples=args.max_samples,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        resume=args.resume,
    )


def run_trio_mode(args: argparse.Namespace) -> None:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key in environment variable {args.api_key_env}.")

    trio = [
        (args.name_a, Path(args.file_a)),
        (args.name_b, Path(args.file_b)),
        (args.name_c, Path(args.file_c)),
    ]

    output_dir = Path(args.output_dir)
    all_summaries: list[dict[str, Any]] = []
    for (name_left, file_left), (name_right, file_right) in itertools.combinations(trio, 2):
        summary = run_pairwise_comparison(
            file_a=file_left,
            file_b=file_right,
            name_a=name_left,
            name_b=name_right,
            judge_model=args.judge_model,
            api_base=args.api_base,
            api_key=api_key,
            output_dir=output_dir,
            seed=args.seed,
            max_samples=args.max_samples,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            resume=args.resume,
        )
        all_summaries.append(summary)

    combined_summary = {
        "mode": "trio",
        "judge_model": args.judge_model,
        "api_base": args.api_base,
        "models": [
            {"name": args.name_a, "file": args.file_a},
            {"name": args.name_b, "file": args.file_b},
            {"name": args.name_c, "file": args.file_c},
        ],
        "pairwise_results": all_summaries,
    }
    save_json(output_dir / "all_pairs.summary.json", combined_summary)
    print(f"[trio] saved combined summary to {output_dir / 'all_pairs.summary.json'}")


def main() -> None:
    args = parse_args()
    if args.command == "pair":
        run_pair_mode(args)
        return
    if args.command == "trio":
        run_trio_mode(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
