from datasets import load_dataset, DatasetDict


def prepare_sampled_dataset(sample_size=5000, seed=42):
    train_full = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    eval_full = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")

    train_sampled = train_full.shuffle(seed=seed).select(range(sample_size))

    final_dataset = DatasetDict({
        "train": train_sampled,
        "test": eval_full
    })

    return final_dataset


if __name__ == "__main__":
    dataset = prepare_sampled_dataset(sample_size=5000)

    train_data = dataset["train"]
    eval_data = dataset["test"]

    print(f"Train size: {len(train_data)}")
    print(f"Eval size: {len(eval_data)}")
    print(dataset)