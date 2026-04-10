from __future__ import annotations

from pathlib import Path
from typing import Tuple

from datasets import Dataset, DatasetDict, load_dataset


REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


def _validate_columns(dataset: Dataset, prompt_col: str, chosen_col: str, rejected_col: str) -> None:
    required = {prompt_col, chosen_col, rejected_col}
    missing = [c for c in required if c not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; available columns: {dataset.column_names}")


def _to_optional_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _standardize(
    record: dict,
    prompt_col: str,
    chosen_col: str,
    rejected_col: str,
    chosen_score_col: str | None,
    rejected_score_col: str | None,
) -> dict:
    return {
        "prompt": str(record[prompt_col]).strip(),
        "chosen": str(record[chosen_col]).strip(),
        "rejected": str(record[rejected_col]).strip(),
        "chosen_score": _to_optional_float(record.get(chosen_score_col)) if chosen_score_col else None,
        "rejected_score": _to_optional_float(record.get(rejected_score_col)) if rejected_score_col else None,
    }


def _extract_dpo_mix(record: dict) -> dict:
    """Convert argilla/dpo-mix-7k message-list fields to plain strings.

    Both *chosen* and *rejected* are lists of ``{"role": ..., "content": ...}``
    dicts.  The prompt is taken from the first user message in *chosen* and
    the response is taken from the last assistant message in each side.
    """
    chosen_msgs = record["chosen"]
    rejected_msgs = record["rejected"]

    # Extract prompt from the first user turn in chosen
    prompt = ""
    if isinstance(chosen_msgs, list):
        for msg in chosen_msgs:
            if msg.get("role") == "user":
                prompt = str(msg["content"]).strip()
                break

    # Extract assistant response from chosen
    if isinstance(chosen_msgs, list) and chosen_msgs:
        chosen = str(chosen_msgs[-1]["content"]).strip()
    else:
        chosen = str(chosen_msgs).strip()

    # Extract assistant response from rejected
    if isinstance(rejected_msgs, list) and rejected_msgs:
        rejected = str(rejected_msgs[-1]["content"]).strip()
    else:
        rejected = str(rejected_msgs).strip()

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_score": _to_optional_float(record.get("chosen_rating")),
        "rejected_score": _to_optional_float(record.get("rejected_rating")),
    }


def load_dpo_datasets(
    use_hub_dataset: bool = True,
    hub_dataset_name: str = "argilla/dpo-mix-7k",
    hub_train_split: str = "train",
    hub_eval_split: str = "test",
    train_file: str = "data/train_dpo.jsonl",
    eval_file: str = "data/eval_dpo.jsonl",
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
    chosen_score_col: str | None = "chosen_rating",
    rejected_score_col: str | None = "rejected_rating",
) -> Tuple[Dataset, Dataset]:
    if use_hub_dataset:
        train_ds = load_dataset(hub_dataset_name, split=hub_train_split)
        eval_ds = load_dataset(hub_dataset_name, split=hub_eval_split)

        train_ds = train_ds.map(
            _extract_dpo_mix,
            remove_columns=train_ds.column_names,
            desc="Processing train split from hub",
        )
        eval_ds = eval_ds.map(
            _extract_dpo_mix,
            remove_columns=eval_ds.column_names,
            desc="Processing eval split from hub",
        )
    else:
        if not Path(train_file).exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        if not Path(eval_file).exists():
            raise FileNotFoundError(f"Eval file not found: {eval_file}")

        loaded = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
        train_ds = loaded["train"]
        eval_ds = loaded["eval"]

        _validate_columns(train_ds, prompt_col, chosen_col, rejected_col)
        _validate_columns(eval_ds, prompt_col, chosen_col, rejected_col)

        train_ds = train_ds.map(
            _standardize,
            fn_kwargs={
                "prompt_col": prompt_col,
                "chosen_col": chosen_col,
                "rejected_col": rejected_col,
                "chosen_score_col": chosen_score_col,
                "rejected_score_col": rejected_score_col,
            },
            remove_columns=train_ds.column_names,
            desc="Standardizing train records",
        )
        eval_ds = eval_ds.map(
            _standardize,
            fn_kwargs={
                "prompt_col": prompt_col,
                "chosen_col": chosen_col,
                "rejected_col": rejected_col,
                "chosen_score_col": chosen_score_col,
                "rejected_score_col": rejected_score_col,
            },
            remove_columns=eval_ds.column_names,
            desc="Standardizing eval records",
        )

    return train_ds, eval_ds


def load_dpo_mix_dataset() -> DatasetDict:
    """Load the full argilla/dpo-mix-7k dataset (train and test splits)."""
    train_ds, eval_ds = load_dpo_datasets()
    return DatasetDict({"train": train_ds, "test": eval_ds})


if __name__ == "__main__":
    dataset = load_dpo_mix_dataset()
    print(f"Train size: {len(dataset['train'])}")
    print(f"Eval size: {len(dataset['test'])}")
    print(dataset)
