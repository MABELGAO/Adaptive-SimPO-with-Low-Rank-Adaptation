# Pairwise LLM-as-a-Judge README

This document explains how to use [scripts/judge_pairwise.py](../scripts/judge_pairwise.py) for pairwise evaluation of model responses.

The script is designed for our DSA5106 project, where we compare `DPO`, `SimPO`, and `Adaptive SimPO` using an `LLM-as-a-Judge` setup.

## What This Script Does

The script compares two model response files on the same prompt set and asks a judge model to decide:

- which response is better
- or whether the result is a `Tie`

It supports:

- a single pairwise comparison: `pair`
- all three pairwise comparisons among three models: `trio`

The intended use case is:

- `DPO vs SimPO`
- `DPO vs Adaptive SimPO`
- `SimPO vs Adaptive SimPO`

## Important Input Rule

The script only reads these three fields from each JSONL row:

- `id`
- `prompt`
- `response`

All other fields are ignored.

This is intentional. Different group members may generate response files with extra metadata fields such as:

- `model_name`
- `adapter_path`
- `generation_config`
- `cleaning_rule`
- `notes`

Those fields are ignored by the judge script so that evaluation remains standardized.

## Expected Input Format

Each input file must be a JSONL file where each line contains at least:

```json
{"id": 0, "prompt": "What is DPO?", "response": "DPO stands for Direct Preference Optimization..."}
```

Requirements:

- `id` must be unique within a file
- the two files being compared must share the same `id` space
- for the same `id`, the `prompt` must match exactly across files

If a row is missing `id`, `prompt`, or `response`, the script raises an error.

If an `id` exists in one file but not the other, that sample is excluded from pairwise judging and reported in the summary.

If the same `id` has different prompts across files, that sample is also excluded and reported in the summary.

## Recommended Response Files

For our project, use cleaned minimal files whenever possible.

For example, for DPO:

- [artifacts/evaluation/alpaca_eval/dpo/alpaca_eval_dpo_responses.minimal.jsonl](../artifacts/evaluation/alpaca_eval/dpo/alpaca_eval_dpo_responses.minimal.jsonl)

This file already contains only:

- `id`
- `prompt`
- `response`

## Judge Model

The script is written for an OpenAI-compatible chat completion API.

Default values:

- judge model: `qwen-max`
- API base: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- API key env var: `OPENAI_API_KEY`

If your deployment uses a different model name or API base, pass them through CLI arguments.

## Randomized Display Order

To reduce position bias, the script randomizes which model is shown as `Response A` and `Response B`.

Important detail:

- the order is deterministic per sample
- it depends on `seed + comparison name + sample id`

This means:

- repeated runs with the same seed are reproducible
- interrupted runs resumed with `--resume` keep the same `A/B` order

## Judge Prompt Behavior

The judge is instructed to compare responses based on:

- correctness and factual accuracy
- relevance
- helpfulness and completeness
- clarity and coherence
- safety

It is also instructed:

- not to prefer a response just because it is longer
- to return `Tie` when both responses are similarly good or similarly flawed

## Output Files

The script writes outputs to `outputs/judgments/` by default.

For a single comparison such as `DPO vs SimPO`, it produces:

- `dpo_vs_simpo.judgments.jsonl`
- `dpo_vs_simpo.summary.json`

For `trio` mode, it additionally produces:

- `all_pairs.summary.json`

## Judgment JSONL Fields

Each row in the judgment file contains analysis-friendly fields such as:

- `id`
- `prompt`
- `model_a`
- `model_b`
- `response_a`
- `response_b`
- `response_a_chars`
- `response_b_chars`
- `response_a_words`
- `response_b_words`
- `length_diff_chars`
- `length_diff_words`
- `display_order`
- `judge_model`
- `judge_winner_label`
- `winner_model`
- `result`
- `judge_reason`
- `judge_raw`

These fields are useful for:

- win/loss/tie statistics
- response-length analysis
- plotting figures for the final report

## Summary JSON Fields

Each summary file includes:

- `wins_a`
- `wins_b`
- `ties`
- `invalid`
- `win_rate_a_excluding_ties`
- `win_rate_b_excluding_ties`
- `adjusted_win_rate_a`
- `adjusted_win_rate_b`
- alignment metadata such as missing ids and prompt mismatches

`adjusted_win_rate` is computed as:

- `(wins + 0.5 * ties) / (wins + losses + ties)`

This is often convenient for plotting or reporting.

## How To Run

### 1. Set the API key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 2. Run one pairwise comparison

```bash
python3 scripts/judge_pairwise.py pair \
  --file-a artifacts/evaluation/alpaca_eval/dpo/alpaca_eval_dpo_responses.minimal.jsonl \
  --file-b path/to/simpo_responses.jsonl \
  --name-a DPO \
  --name-b SimPO \
  --judge-model qwen-max \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --output-dir outputs/judgments \
  --seed 42
```

### 3. Run a small test first

```bash
python3 scripts/judge_pairwise.py pair \
  --file-a artifacts/evaluation/alpaca_eval/dpo/alpaca_eval_dpo_responses.minimal.jsonl \
  --file-b path/to/simpo_responses.jsonl \
  --name-a DPO \
  --name-b SimPO \
  --max-samples 10 \
  --judge-model qwen-max \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --output-dir outputs/judgments \
  --seed 42
```

This is recommended before full evaluation so we can inspect:

- whether the API works
- whether the judge output is stable
- whether `judge_reason` looks sensible

### 4. Run all three pairwise comparisons

```bash
python3 scripts/judge_pairwise.py trio \
  --file-a artifacts/evaluation/alpaca_eval/dpo/alpaca_eval_dpo_responses.minimal.jsonl \
  --file-b path/to/simpo.jsonl \
  --file-c path/to/adaptive_simpo.jsonl \
  --name-a DPO \
  --name-b SimPO \
  --name-c AdaptiveSimPO \
  --judge-model qwen-max \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --output-dir outputs/judgments \
  --seed 42
```

### 5. Resume after interruption

```bash
python3 scripts/judge_pairwise.py pair \
  --file-a artifacts/evaluation/alpaca_eval/dpo/alpaca_eval_dpo_responses.minimal.jsonl \
  --file-b path/to/simpo_responses.jsonl \
  --name-a DPO \
  --name-b SimPO \
  --judge-model qwen-max \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --output-dir outputs/judgments \
  --seed 42 \
  --resume
```

## Notes on Invalid Judgments

The script expects the judge model to return JSON:

```json
{"winner":"A|B|Tie","reason":"short explanation"}
```

If the model returns malformed output, the script applies a limited fallback parser.

The fallback is intentionally conservative:

- it accepts very clear patterns like `Winner: B`
- it accepts clear tie-like statements such as `Both responses are about equally good`
- it marks ambiguous cases as `Invalid`

This is safer than aggressively guessing `A` or `B`, because silent misclassification would contaminate evaluation results.

## Recommended Workflow For Our Group

1. Generate response files for each model.
2. Clean them to a minimal `id/prompt/response` format if necessary.
3. Run a small `--max-samples 10` test.
4. Inspect the `.judgments.jsonl` file.
5. Run the full pairwise or trio comparison.
6. Use the summary JSON and judgments JSONL for tables, plots, and report analysis.

## Guidance For AI Assistants

If another AI assistant reads this repo and needs to help with evaluation, it should follow these rules:

- Do not assume extra metadata fields are usable.
- Only rely on `id`, `prompt`, and `response` from response files.
- Do not reorder ids manually unless explicitly required.
- Preserve the existing output format so downstream analysis scripts remain compatible.
- Prefer creating new output files rather than overwriting existing judgments without permission.
- Treat `invalid` judgments as a visible quality-control signal, not as noise to silently discard.

## Related Files

- Judge script: [scripts/judge_pairwise.py](../scripts/judge_pairwise.py)
- Cleaning script: [scripts/clean_alpaca_eval_responses.py](../scripts/clean_alpaca_eval_responses.py)
- DPO minimal response file: [artifacts/evaluation/alpaca_eval/dpo/alpaca_eval_dpo_responses.minimal.jsonl](../artifacts/evaluation/alpaca_eval/dpo/alpaca_eval_dpo_responses.minimal.jsonl)
- Raw SimPO inference output: [artifacts/evaluation/alpaca_eval/simpo/simpo_alpaca_inference_outputs.json](../artifacts/evaluation/alpaca_eval/simpo/simpo_alpaca_inference_outputs.json)
