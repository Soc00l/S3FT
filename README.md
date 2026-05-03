# S3FT Generalization Project

Reproduction and extension project for `Selective Self-to-Supervised Fine-Tuning for Generalization in Large Language Models`.

This repo is organized around one core question:

`Does standard SFT overfit to a single gold answer, and can selectively training on model-generated correct answers preserve better generalization and downstream plasticity?`

## Scope

The first project milestone focuses on a math-reasoning setup:

- Task: `GSM8K -> GSM8K / SVAMP / MATH subset`
- Models: `Qwen2.5-1.5B` or `Qwen2.5-3B`
- Baselines:
  - `Vanilla SFT`
  - `S3FT`
  - `Self-answer only`
- Judge: exact numeric matching with normalization

## Repository Layout

```text
configs/         Training and experiment configs
data/            Raw data and rewritten datasets
docs/research/   Reading notes, trend summaries, and project plans
reports/         Experiment logs and case studies
results/         Metrics, tables, and exported analysis artifacts
scripts/         Data prep, training, generation, and evaluation scripts
```

## Planned Pipeline

1. Prepare `GSM8K` in a unified instruction format.
2. Train a `Vanilla SFT` baseline.
3. Generate model self-answers on the training set.
4. Judge whether each self-answer is correct.
5. Rewrite the training set into an `S3FT` version.
6. Retrain and compare against baselines.
7. Analyze generalization, judge quality, and test-time scaling behavior.

## First Deliverables

- Reproducible baseline SFT training config
- Self-answer generation script
- Numeric judge for math tasks
- S3FT dataset builder
- Unified evaluation script for `GSM8K`, `SVAMP`, and `MATH subset`

## Research Notes

- [2024-2025 post-training paper map](./docs/research/post_training_a_conf_papers_2024_2025.md)
- [2025 post-training trend map](./docs/research/post_training_trends_2025.md)
- [ILR / SPPO / RISE / S3FT project evaluation](./docs/research/project_eval_ilr_sppo_rise_s3ft.md)
- [S3FT project plan](./docs/research/s3ft_project_plan.md)
- [S3FT execution checklist](./docs/research/s3ft_execution_checklist.md)

## Setup

Create a Python environment and install the dependencies in `requirements.txt`.

## Status

Project scaffold initialized. Training and data scripts are being built out from the math-reasoning baseline first.
