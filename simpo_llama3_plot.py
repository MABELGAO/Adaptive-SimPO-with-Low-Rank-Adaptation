import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl.experimental.cpo import CPOTrainer, CPOConfig

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

BEST_LR = 8e-7
BEST_GAMMA = 1.4
BEST_BETA = 2.0
CPO_ALPHA = 0.0

MAX_PROMPT_LENGTH = 1024
MAX_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 64
EPOCHS = 1
LORA_R = 16
LORA_ALPHA = 32

LOGGING_STEPS = 1
EVAL_STEPS = 50

FORMAL_OUT_DIR = "./llama-3-8b-instruct-simpo-final"
METRICS_OUT_DIR = "./simpo_metrics_output"


def setup_directories():
    os.makedirs(FORMAL_OUT_DIR, exist_ok=True)
    os.makedirs(METRICS_OUT_DIR, exist_ok=True)


def prepare_full_dataset():
    dataset = load_dataset("argilla/dpo-mix-7k", split="train")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({
        "train": split_dataset["train"],
        "test": split_dataset["test"]
    })


def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    return model, tokenizer


class ResourceLoggingCallback(TrainerCallback):
    def __init__(self):
        self.step_start_time = 0
        self.resource_data = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.step_start_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        self.resource_data.append({
            "step": state.global_step,
            "time_per_step_s": step_time,
            "peak_gpu_memory_gb": peak_mem
        })


def run_formal_training(dataset):
    model, tokenizer = get_model_and_tokenizer()

    args = CPOConfig(
        output_dir=FORMAL_OUT_DIR,
        learning_rate=BEST_LR,
        beta=BEST_BETA,
        simpo_gamma=BEST_GAMMA,
        cpo_alpha=CPO_ALPHA,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch",
        bf16=True,
        report_to="none"
    )

    trainer = CPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH
    )

    resource_callback = ResourceLoggingCallback()
    trainer.add_callback(resource_callback)

    start_time = time.time()
    trainer.train()
    training_time_minutes = (time.time() - start_time) / 60.0

    trainer.save_model(FORMAL_OUT_DIR)
    tokenizer.save_pretrained(FORMAL_OUT_DIR)

    return trainer.state.log_history, resource_callback.resource_data, training_time_minutes


def format_plot(ax, title, ylabel):
    ax.set_title(title, fontsize=16, fontweight='bold', loc='left', pad=15)
    ax.set_xlabel('Training Step', fontweight='bold', fontsize=12)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12, rotation=0, loc='top', labelpad=20)
    ax.grid(color='#D8DEE6', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.set_facecolor('#FBFCFE')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2D3748')
    ax.spines['bottom'].set_color('#2D3748')
    ax.legend(frameon=False, loc='upper right')


def extract_and_plot_metrics(log_history, resource_data):
    metrics = {
        "train_loss": [], "eval_loss": [],
        "train_margin": [], "eval_margin": [],
        "train_acc": [], "eval_acc": [],
        "chosen_len": [], "rejected_len": [], "len_diff": []
    }

    for log in log_history:
        step = log.get("step")
        if step is None: continue

        if "loss" in log and "eval_loss" not in log:
            metrics["train_loss"].append({"step": step, "value": log["loss"]})
        if "eval_loss" in log:
            metrics["eval_loss"].append({"step": step, "value": log["eval_loss"]})

        if "rewards/margins" in log:
            metrics["train_margin"].append({"step": step, "value": log["rewards/margins"]})
        if "eval_rewards/margins" in log:
            metrics["eval_margin"].append({"step": step, "value": log["eval_rewards/margins"]})

        if "rewards/accuracies" in log:
            metrics["train_acc"].append({"step": step, "value": log["rewards/accuracies"]})
        if "eval_rewards/accuracies" in log:
            metrics["eval_acc"].append({"step": step, "value": log["eval_rewards/accuracies"]})

        if "rewards/chosen_lengths" in log and "rewards/rejected_lengths" in log:
            c_len = log["rewards/chosen_lengths"]
            r_len = log["rewards/rejected_lengths"]
            metrics["chosen_len"].append({"step": step, "value": c_len})
            metrics["rejected_len"].append({"step": step, "value": r_len})
            metrics["len_diff"].append({"step": step, "value": c_len - r_len})

    for key, data in metrics.items():
        if data:
            pd.DataFrame(data).to_csv(os.path.join(METRICS_OUT_DIR, f"{key}.csv"), index=False)

    if resource_data:
        pd.DataFrame(resource_data).to_csv(os.path.join(METRICS_OUT_DIR, "runtime_memory.csv"), index=False)

    def get_xy(data_list):
        if not data_list: return [], []
        return [d["step"] for d in data_list], [d["value"] for d in data_list]

    # --- 1. Loss Curve ---
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#FBFCFE')
    tx, ty = get_xy(metrics["train_loss"])
    ex, ey = get_xy(metrics["eval_loss"])
    if tx: ax.plot(tx, ty, color='#1D4ED8', label='Train Loss', linewidth=2, marker='.', markersize=6)
    if ex: ax.plot(ex, ey, color='#DC2626', label='Eval Loss', linewidth=2, marker='o', markersize=8)
    format_plot(ax, 'Loss Curve', 'Loss')
    plt.savefig(os.path.join(METRICS_OUT_DIR, "loss_curve.png"), bbox_inches='tight')
    plt.close()

    # --- 2. Reward Dynamics ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    fig.patch.set_facecolor('#FBFCFE')
    tx, ty = get_xy(metrics["train_margin"])
    ex, ey = get_xy(metrics["eval_margin"])
    if tx: axs[0].plot(tx, ty, color='#7C3AED', label='Train Margin', linewidth=2, marker='.', markersize=6)
    if ex: axs[0].plot(ex, ey, color='#EA580C', label='Eval Margin', linewidth=2, marker='o', markersize=8)
    format_plot(axs[0], 'Reward Margin', 'Margin')

    tx, ty = get_xy(metrics["train_acc"])
    ex, ey = get_xy(metrics["eval_acc"])
    if tx: axs[1].plot(tx, ty, color='#059669', label='Train Accuracy', linewidth=2, marker='.', markersize=6)
    if ex: axs[1].plot(ex, ey, color='#B91C1C', label='Eval Accuracy', linewidth=2, marker='o', markersize=8)
    format_plot(axs[1], 'Reward Accuracy', 'Accuracy')
    plt.savefig(os.path.join(METRICS_OUT_DIR, "reward_metrics.png"), bbox_inches='tight')
    plt.close()

    # --- 3. Response Length Behavior ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    fig.patch.set_facecolor('#FBFCFE')
    cx, cy = get_xy(metrics["chosen_len"])
    rx, ry = get_xy(metrics["rejected_len"])
    if cx: axs[0].plot(cx, cy, color='#2563EB', label='Chosen', linewidth=2, marker='.', markersize=6)
    if rx: axs[0].plot(rx, ry, color='#DB2777', label='Rejected', linewidth=2, marker='.', markersize=6)
    format_plot(axs[0], 'Chosen vs Rejected Length', 'Tokens')

    dx, dy = get_xy(metrics["len_diff"])
    if dx: axs[1].plot(dx, dy, color='#0F766E', label='Length Diff', linewidth=2, marker='.', markersize=6)
    format_plot(axs[1], 'Length Difference\nChosen - Rejected', '')
    plt.savefig(os.path.join(METRICS_OUT_DIR, "response_length_metrics.png"), bbox_inches='tight')
    plt.close()

    # --- 4. Runtime and Memory ---
    if resource_data:
        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        fig.patch.set_facecolor('#FBFCFE')
        rx = [d["step"] for d in resource_data]
        rt = [d["time_per_step_s"] for d in resource_data]
        rm = [d["peak_gpu_memory_gb"] for d in resource_data]

        axs[0].plot(rx, rt, color='#9333EA', label='Step Time', linewidth=2, marker='.', markersize=6)
        format_plot(axs[0], 'Time Per Step', 'Seconds')

        axs[1].plot(rx, rm, color='#B45309', label='Peak GPU Memory', linewidth=2, marker='.', markersize=6)
        format_plot(axs[1], 'Peak GPU Memory', 'GB')

        plt.savefig(os.path.join(METRICS_OUT_DIR, "runtime_memory_metrics.png"), bbox_inches='tight')
        plt.close()


def main():
    setup_directories()
    dataset = prepare_full_dataset()
    log_history, resource_data, training_time = run_formal_training(dataset)
    extract_and_plot_metrics(log_history, resource_data)


if __name__ == "__main__":
    main()