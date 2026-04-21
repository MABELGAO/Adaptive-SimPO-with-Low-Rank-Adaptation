# Adaptive SimPO with Low-Rank Adaptation

This repository contains the implementation for the **DSA5106** Project, focusing on the reproduction and extension of the SimPO 
(Simple Preference Optimization) algorithm. Our work aims to bridge the gap between state-of-the-art LLM alignment and accessible 
computation by utilizing QLoRA and introducing an Adaptive Margin mechanism.


## Overview

- Reproduce **SimPO** (reference-free preference optimization)
- Compare against **DPO (Direct Preference Optimization)**
- Propose an extension: **Adaptive Margin SimPO**
- Use **QLoRA** for efficient training on limited hardware
- Use qwen3-max serving as the **LLM judge**

## Key Ideas

- SimPO removes the need for a reference model → reduces memory usage
- Addresses length bias in preference optimization
- Our extension dynamically adjusts margin based on preference gap

## Repository Structure
The project is organized into functional branches to manage the baseline development and our novel extension:

main: The stable branch containing README documentation and data.

SimPO: Implementation of the standard SimPO loss function and reproduction experiments.

DPO: Baseline implementation of Direct Preference Optimization for performance benchmarking.

SimPOextension: Development of the Adaptive Margin SimPO, including the sigmoid-based nonlinear mapping logic.

## Metrics:
Win Rate (LLM-as-a-Judge)
Response Length Analysis
Training Efficiency (VRAM, throughput)

## Models & Data
Base Model: LLaMA-3-8B / Mistral-7B
Dataset: UltraFeedback (subset)
Method: QLoRA (4-bit)

## 👥 Team
Yang Lujianing
Zhao Yiming
Gao Qiman
Su Siying
Tang Zijun
Cao Zhizhou
Yan Junjie

## References
[1] Meng, Y., et al. (2024). SimPO: Simple Preference Optimization with a Reference-Free Reward. NeurIPS 2024.

[2] Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model.

[3] Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.

