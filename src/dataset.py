from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset


REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


def prepare_sampled_dataset(sample_size: int = 5000, seed: int = 42) -> DatasetDict:
    """Load and sample the ultrafeedback_binarized dataset.

    Shuffles the training split and selects *sample_size* examples while
    keeping the full evaluation split, exactly as required by the problem
    specification.
    """
    train_full = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    eval_full = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")

    train_sampled = train_full.shuffle(seed=seed).select(range(sample_size))

    final_dataset = DatasetDict({
        "train": train_sampled,
        "test": eval_full,
    })

    return final_dataset


def _validate_columns(dataset: Dataset, prompt_col: str, chosen_col: str, rejected_col: str) -> None:
    required = {prompt_col, chosen_col, rejected_col}
    missing = [c for c in required if c not in dataset.column_names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; available columns: {dataset.column_names}")


def _standardize(record: dict, prompt_col: str, chosen_col: str, rejected_col: str) -> dict:
    return {
        "prompt": str(record[prompt_col]).strip(),
        "chosen": str(record[chosen_col]).strip(),
        "rejected": str(record[rejected_col]).strip(),
    }


def _extract_ultrafeedback(record: dict) -> dict:
    """Convert ultrafeedback_binarized message-list fields to plain strings."""
    prompt = str(record["prompt"]).strip()

    chosen_msgs = record["chosen"]
    if isinstance(chosen_msgs, list) and chosen_msgs:
        chosen = str(chosen_msgs[-1]["content"]).strip()
    else:
        chosen = str(chosen_msgs).strip()

    rejected_msgs = record["rejected"]
    if isinstance(rejected_msgs, list) and rejected_msgs:
        rejected = str(rejected_msgs[-1]["content"]).strip()
    else:
        rejected = str(rejected_msgs).strip()

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def load_dpo_datasets(
    use_hub_dataset: bool = True,
    hub_dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    hub_train_split: str = "train_prefs",
    hub_eval_split: str = "test_prefs",
    train_file: str = "data/train_dpo.jsonl",
    eval_file: str = "data/eval_dpo.jsonl",
    prompt_col: str = "prompt",
    chosen_col: str = "chosen",
    rejected_col: str = "rejected",
    max_train_samples: Optional[int] = 5000,
    max_eval_samples: Optional[int] = None,
    shuffle_train: bool = True,
    train_sample_seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    if use_hub_dataset:
        train_ds = load_dataset(hub_dataset_name, split=hub_train_split)
        eval_ds = load_dataset(hub_dataset_name, split=hub_eval_split)

        train_ds = train_ds.map(
            _extract_ultrafeedback,
            remove_columns=train_ds.column_names,
            desc="Processing train split from hub",
        )
        eval_ds = eval_ds.map(
            _extract_ultrafeedback,
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
            },
            remove_columns=eval_ds.column_names,
            desc="Standardizing eval records",
        )

    if max_train_samples is not None:
        if shuffle_train:
            train_ds = train_ds.shuffle(seed=train_sample_seed)
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))

    return train_ds, eval_ds


_DEFAULT_SAMPLE_SIZE = 5000

if __name__ == "__main__":
    dataset = prepare_sampled_dataset(sample_size=_DEFAULT_SAMPLE_SIZE)
    train_data = dataset["train"]
    eval_data = dataset["test"]
    print(f"Train size: {len(train_data)}")
    print(f"Eval size: {len(eval_data)}")
    print(dataset)
