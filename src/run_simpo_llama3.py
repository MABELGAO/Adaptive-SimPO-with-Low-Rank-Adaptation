# --- IMPORTS ---
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, DatasetDict
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl.experimental.cpo import CPOTrainer, CPOConfig

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- GLOBAL CONFIGURATIONS ---
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_ID = "/root/autodl-tmp/llama3_model/Llama3_8B"

TUNE_GAMMAS = [1.0, 1.4]
TUNE_LRS = [5e-7, 8e-7]
TUNE_STEPS = 50

BEST_LR = 8e-7
BEST_GAMMA = 1.4
BEST_BETA = 2.0
CPO_ALPHA = 0.0

MAX_PROMPT_LENGTH = 1024
MAX_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
EPOCHS = 1
LORA_R = 16
LORA_ALPHA = 32

TUNE_OUT_DIR = "./results/simpo_tuning_results"
FORMAL_OUT_DIR = "./results/llama-3-8b-instruct-simpo-final"
METRICS_OUT_DIR = "./results/simpo_metrics_output"


# --- DIRECTORY INITIALIZATION ---
def setup_directories():
    os.makedirs(TUNE_OUT_DIR, exist_ok=True)
    os.makedirs(FORMAL_OUT_DIR, exist_ok=True)
    os.makedirs(METRICS_OUT_DIR, exist_ok=True)


# --- DATASET PIPELINE ---
# def prepare_full_dataset():
#     dataset = load_dataset("argilla/dpo-mix-7k", split="train[:2000]")
#     split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

#     final_dataset = DatasetDict({
#         "train": split_dataset["train"],
#         "test": split_dataset["test"]
#     })
#     return final_dataset
def prepare_full_dataset():
    # full_dataset = load_from_disk("/root/autodl-tmp/project/dpo_mix_7k")
    # dataset = full_dataset["train"].select(range(2000))
    dataset = load_from_disk("/root/autodl-tmp/project/dpo_mix_7k")
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
                loss_type="simpo",
                learning_rate=lr,
                beta=BEST_BETA,
                simpo_gamma=gamma,
                cpo_alpha=CPO_ALPHA,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM_STEPS,
                max_steps=TUNE_STEPS,
                # max_prompt_length=MAX_PROMPT_LENGTH,
                max_length=MAX_LENGTH,
                eval_strategy="steps",
                eval_steps=10,
                logging_steps=5,
                optim="adamw_torch",
                bf16=True,
                report_to="none",
                remove_unused_columns=False,
                gradient_checkpointing=True, # 必须开启，保护显存不产生瞬时峰值
                # bf16=True,                   # 5090 对 bf16 的原生支持极强
                tf32=True,                   # 开启 TF32 加速矩阵运算
                eval_accumulation_steps=1
            )

            trainer = CPOTrainer(
                model=model,
                args=args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                # tokenizer=tokenizer
                processing_class=tokenizer
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
        loss_type="simpo",
        learning_rate=BEST_LR,
        beta=BEST_BETA,
        simpo_gamma=BEST_GAMMA,
        cpo_alpha=CPO_ALPHA,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        # max_prompt_length=MAX_PROMPT_LENGTH,
        max_length=MAX_LENGTH,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=20,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch",
        bf16=True,
        report_to="none",
        max_grad_norm=1.0,
        gradient_checkpointing=True, # 必须开启，保护显存不产生瞬时峰值
        # bf16=True,                   # 5090 对 bf16 的原生支持极强
        tf32=True,                   # 开启 TF32 加速矩阵运算
        # per_device_eval_batch_size=1, # 极其重要：将评估 Batch 降到最低
        eval_accumulation_steps=1, # 极其重要：每步评估后立刻清理显存，不堆积
    )

    trainer = CPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # tokenizer=tokenizer
        processing_class=tokenizer
    )

    start_time = time.time()
    trainer.train()

    all_keys = set()
    for row in trainer.state.log_history:
        all_keys.update(row.keys())

    print("\n=== ALL LOG KEYS ===")
    print(sorted(all_keys))

    end_time = time.time()

    training_time_minutes = (end_time - start_time) / 60.0

    trainer.save_model(FORMAL_OUT_DIR)
    tokenizer.save_pretrained(FORMAL_OUT_DIR)

    print("\n=== LOG HISTORY LENGTH ===")
    print(len(trainer.state.log_history))

    print("\n=== FULL LOG HISTORY ===")
    for row in trainer.state.log_history:
        print(row)

    avg_response_length = compute_avg_response_length(dataset["test"], tokenizer)

    return trainer.state.log_history, training_time_minutes, avg_response_length, model, tokenizer

# --- METRICS EXTRACTION AND VISUALIZATION ---
def extract_and_plot_metrics(
    log_history,
    training_time,
    avg_response_length=None,
    eval_dataset=None,
    model=None,
    tokenizer=None
):
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

        # if "eval_rewards/chosen_lengths" in log and "eval_rewards/rejected_lengths" in log:
        #     avg_len = (log["eval_rewards/chosen_lengths"] + log["eval_rewards/rejected_lengths"]) / 2.0
        #     length_data.append({"step": step, "avg_response_length": avg_len})
        if avg_response_length is not None:
            length_data.append({"step": step, "avg_response_length": avg_response_length})

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
        axs[0, 0].plot(df_loss['step'], df_loss['training_loss'], color='red', marker='o')
        axs[0, 0].set_title('1. Training Loss')
        axs[0, 0].set_xlabel('Steps')
        axs[0, 0].grid(True)

    if not df_margin.empty:
        axs[0, 1].plot(df_margin['step'], df_margin['reward_margin'], color='blue', marker='o')
        axs[0, 1].set_title('2. Reward Margin')
        axs[0, 1].set_xlabel('Steps')
        axs[0, 1].grid(True)

    if not df_len.empty:
        axs[1, 0].plot(df_len['step'], df_len['avg_response_length'], color='green', marker='o')
        axs[1, 0].set_title('4. Response Length')
        axs[1, 0].set_xlabel('Steps')
        axs[1, 0].grid(True)

    if not df_acc.empty:
        axs[1, 1].plot(df_acc['step'], df_acc['reward_accuracy'], color='purple', marker='o')
        axs[1, 1].set_title('5. Reward Accuracy')
        axs[1, 1].set_xlabel('Steps')
        axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_OUT_DIR, "formal_training_metrics.png"))
    plt.close()

        # --- SimPO-style scatter: response length vs avg log prob ---
    if eval_dataset is not None and model is not None and tokenizer is not None:
        df_scatter, rho = compute_length_logprob_scatter_data(
            dataset=eval_dataset,
            model=model,
            tokenizer=tokenizer,
            max_samples=1000,
            out_dir=METRICS_OUT_DIR
        )

        df_scatter.to_csv(
            os.path.join(METRICS_OUT_DIR, "7_length_vs_avg_logprob.csv"),
            index=False
        )

        plt.figure(figsize=(8, 6))
        plt.scatter(df_scatter["response_length"], df_scatter["avg_log_prob"], s=8, alpha=0.6)
        plt.xlabel(r"Response length $|y|$")
        plt.ylabel(r"Avg. log prob. $p_\theta(y \mid x)$")
        plt.title("SimPO-style Length vs Avg. Log Prob.")
        plt.text(
            0.78, 0.12,
            rf"$\rho = {rho:.2f}$",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )
        plt.tight_layout()
        plt.savefig(os.path.join(METRICS_OUT_DIR, "7_length_vs_avg_logprob.png"))
        plt.close()

        print("\n=== LENGTH vs AVG LOGPROB ===")
        print(df_scatter.head())
        print(f"Correlation rho = {rho:.4f}")


# --- RESPONSE LENGTH ---
def compute_avg_response_length(dataset, tokenizer):
    chosen_lens = []
    rejected_lens = []

    for ex in dataset:
        # chosen / rejected are lists like:
        # [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        chosen_msg = ex["chosen"][-1]["content"]
        rejected_msg = ex["rejected"][-1]["content"]

        chosen_ids = tokenizer(chosen_msg, add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(rejected_msg, add_special_tokens=False)["input_ids"]

        chosen_lens.append(len(chosen_ids))
        rejected_lens.append(len(rejected_ids))

    if len(chosen_lens) == 0:
        return None

    return (sum(chosen_lens) + sum(rejected_lens)) / (len(chosen_lens) + len(rejected_lens))

def compute_length_logprob_scatter_data(dataset, model, tokenizer, out_dir, max_samples=1000):
    """
    Reproduce a SimPO-style scatter plot:
    x-axis: response length |y|
    y-axis: average log-probability of the response under the current model

    We use the chosen response by default.
    """
    model.eval()

    lengths = []
    avg_logps = []

    # 如果样本太多，可以限制数量，避免分析太慢
    n = min(len(dataset), max_samples)

    for i in range(n):
        ex = dataset[i]

        # 数据格式：list of messages
        prompt_msgs = ex["chosen"][:-1]      # user / context
        response_msg = ex["chosen"][-1]      # assistant response

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        response_text = response_msg["content"]

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        response_ids = tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 语言模型预测第 t 个 token 用的是第 t-1 个位置的 logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            log_probs = torch.log_softmax(shift_logits, dim=-1)

            # 取出真实 token 的 log-prob
            token_logps = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            prompt_len = prompt_ids.size(1)
            response_len = response_ids.size(1)

            # token_logps 对应的是 input_ids[:, 1:]
            # response 部分对应的起点是 prompt_len-1
            start = prompt_len - 1
            end = start + response_len

            response_token_logps = token_logps[:, start:end]

            total_logp = response_token_logps.sum().item()
            avg_logp = total_logp / response_len

        lengths.append(response_len)
        avg_logps.append(avg_logp)

    df_scatter = pd.DataFrame({
        "response_length": lengths,
        "avg_log_prob": avg_logps
    })
    df_scatter.to_csv(os.path.join(out_dir, "7_length_vs_avg_logprob.csv"), index=False)

    # Pearson correlation
    rho = np.corrcoef(lengths, avg_logps)[0, 1] if len(lengths) > 1 else np.nan

    plt.figure(figsize=(8, 6))
    plt.scatter(lengths, avg_logps, s=8, alpha=0.6)
    plt.xlabel(r"Response length $|y|$")
    plt.ylabel(r"Avg. log prob. $p_\theta(y \mid x)$")
    plt.title("SimPO-style Length vs Avg. Log Prob.")
    plt.text(
        0.78, 0.12,
        rf"$\rho = {rho:.2f}$",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "7_length_vs_avg_logprob.png"))
    plt.close()

    return df_scatter, rho

# --- MAIN EXECUTION ---
def main():
    setup_directories()
    dataset = prepare_full_dataset()

    # run_tuning_phase(dataset)

    # log_history, training_time, avg_response_length = run_formal_training(dataset)

    # extract_and_plot_metrics(log_history, training_time, avg_response_length)
    log_history, training_time, avg_response_length, model, tokenizer = run_formal_training(dataset)
    extract_and_plot_metrics(log_history, training_time, avg_response_length, dataset["test"], model, tokenizer)


if __name__ == "__main__":
    main()
