# --- IMPORTS ---
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import CPOTrainer, CPOConfig

# --- GLOBAL CONFIGURATIONS ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

TUNE_GAMMAS = [1.0, 1.4]
TUNE_LRS = [5e-7, 8e-7]
TUNE_STEPS = 50

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

TUNE_OUT_DIR = "./simpo_tuning_results"
FORMAL_OUT_DIR = "./llama-3-8b-instruct-simpo-final"
METRICS_OUT_DIR = "./simpo_metrics_output"


# --- DIRECTORY INITIALIZATION ---
def setup_directories():
    os.makedirs(TUNE_OUT_DIR, exist_ok=True)
    os.makedirs(FORMAL_OUT_DIR, exist_ok=True)
    os.makedirs(METRICS_OUT_DIR, exist_ok=True)


# --- DATASET PIPELINE ---
def prepare_full_dataset():
    dataset = load_dataset("argilla/dpo-mix-7k", split="train")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    final_dataset = DatasetDict({
        "train": split_dataset["train"],
        "test": split_dataset["test"]
    })
    return final_dataset


# --- MODEL AND TOKENIZER SETUP ---
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


# --- HYPERPARAMETER TUNING PHASE ---
def run_tuning_phase(dataset):
    tuning_results = []

    for gamma in TUNE_GAMMAS:
        for lr in TUNE_LRS:
            model, tokenizer = get_model_and_tokenizer()

            run_name = f"tune_gamma_{gamma}_lr_{lr}"
            out_path = os.path.join(TUNE_OUT_DIR, run_name)

            args = CPOConfig(
                output_dir=out_path,
                learning_rate=lr,
                beta=BEST_BETA,
                simpo_gamma=gamma,
                cpo_alpha=CPO_ALPHA,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM_STEPS,
                max_steps=TUNE_STEPS,
                max_prompt_length=MAX_PROMPT_LENGTH,
                max_length=MAX_LENGTH,
                eval_strategy="steps",
                eval_steps=10,
                logging_steps=10,
                optim="adamw_torch",
                bf16=True,
                report_to="none"
            )

            trainer = CPOTrainer(
                model=model,
                args=args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                tokenizer=tokenizer
            )

            trainer.train()

            history = trainer.state.log_history
            final_margin = 0
            final_loss = 0

            for log in reversed(history):
                if "eval_rewards/margins" in log:
                    final_margin = log["eval_rewards/margins"]
                if "loss" in log:
                    final_loss = log["loss"]
                if final_margin != 0 and final_loss != 0:
                    break

            tuning_results.append({
                "gamma": gamma,
                "learning_rate": lr,
                "final_train_loss": final_loss,
                "final_eval_margin": final_margin
            })

            del trainer
            del model
            torch.cuda.empty_cache()

    df_tune = pd.DataFrame(tuning_results)
    df_tune.to_csv(os.path.join(TUNE_OUT_DIR, "tuning_results.csv"), index=False)

    plt.figure(figsize=(10, 6))
    for gamma in TUNE_GAMMAS:
        subset = df_tune[df_tune['gamma'] == gamma]
        plt.plot(subset['learning_rate'], subset['final_eval_margin'], marker='o', label=f'Gamma={gamma}')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Eval Margin')
    plt.title('SimPO Tuning: Margin vs Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(TUNE_OUT_DIR, "tuning_plot.png"))
    plt.close()


# --- FORMAL TRAINING PHASE ---
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
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_length=MAX_LENGTH,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
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
        tokenizer=tokenizer
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time_minutes = (end_time - start_time) / 60.0

    trainer.save_model(FORMAL_OUT_DIR)
    tokenizer.save_pretrained(FORMAL_OUT_DIR)

    return trainer.state.log_history, training_time_minutes


# --- METRICS EXTRACTION AND VISUALIZATION ---
def extract_and_plot_metrics(log_history, training_time):
    train_loss_data = []
    margin_data = []
    accuracy_data = []
    length_data = []

    for log in log_history:
        step = log.get("step")
        if "loss" in log and "eval_loss" not in log:
            train_loss_data.append({"step": step, "training_loss": log["loss"]})

        if "eval_rewards/margins" in log:
            margin_data.append({"step": step, "reward_margin": log["eval_rewards/margins"]})

        if "eval_rewards/accuracies" in log:
            accuracy_data.append({"step": step, "reward_accuracy": log["eval_rewards/accuracies"]})

        if "eval_rewards/chosen_lengths" in log and "eval_rewards/rejected_lengths" in log:
            avg_len = (log["eval_rewards/chosen_lengths"] + log["eval_rewards/rejected_lengths"]) / 2.0
            length_data.append({"step": step, "avg_response_length": avg_len})

    df_loss = pd.DataFrame(train_loss_data)
    df_margin = pd.DataFrame(margin_data)
    df_acc = pd.DataFrame(accuracy_data)
    df_len = pd.DataFrame(length_data)

    df_loss.to_csv(os.path.join(METRICS_OUT_DIR, "1_training_loss.csv"), index=False)
    df_margin.to_csv(os.path.join(METRICS_OUT_DIR, "2_reward_margin.csv"), index=False)
    df_acc.to_csv(os.path.join(METRICS_OUT_DIR, "5_reward_accuracy.csv"), index=False)
    df_len.to_csv(os.path.join(METRICS_OUT_DIR, "4_response_length.csv"), index=False)

    time_data = [{"metric": "training_time_minutes", "value": training_time}]
    pd.DataFrame(time_data).to_csv(os.path.join(METRICS_OUT_DIR, "6_training_time.csv"), index=False)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    if not df_loss.empty:
        axs[0, 0].plot(df_loss['step'], df_loss['training_loss'], color='red')
        axs[0, 0].set_title('1. Training Loss')
        axs[0, 0].set_xlabel('Steps')
        axs[0, 0].grid(True)

    if not df_margin.empty:
        axs[0, 1].plot(df_margin['step'], df_margin['reward_margin'], color='blue')
        axs[0, 1].set_title('2. Reward Margin')
        axs[0, 1].set_xlabel('Steps')
        axs[0, 1].grid(True)

    if not df_len.empty:
        axs[1, 0].plot(df_len['step'], df_len['avg_response_length'], color='green')
        axs[1, 0].set_title('4. Response Length')
        axs[1, 0].set_xlabel('Steps')
        axs[1, 0].grid(True)

    if not df_acc.empty:
        axs[1, 1].plot(df_acc['step'], df_acc['reward_accuracy'], color='purple')
        axs[1, 1].set_title('5. Reward Accuracy')
        axs[1, 1].set_xlabel('Steps')
        axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_OUT_DIR, "formal_training_metrics.png"))
    plt.close()


# --- MAIN EXECUTION ---
def main():
    setup_directories()
    dataset = prepare_full_dataset()

    run_tuning_phase(dataset)

    log_history, training_time = run_formal_training(dataset)

    extract_and_plot_metrics(log_history, training_time)


if __name__ == "__main__":
    main()