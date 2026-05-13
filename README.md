# S3FT 泛化复现与扩展

这是一个围绕论文 `Selective Self-to-Supervised Fine-Tuning for Generalization in Large Language Models` 的复现与扩展项目。

项目当前聚焦一个更像研究实验的核心问题：

`单一 gold answer 的监督会不会让模型变窄？如果选择性地引入模型自己生成的正确答案，能不能更好地保留泛化能力？`

## 当前实验设定

- 任务：`GSM8K -> GSM8K / SVAMP`
- 底模：`Qwen2.5-1.5B-Instruct`
- 方法对比：
  - `Vanilla SFT`
  - `S3FT`
- judge：基于数值归一化的 exact match

当前主实验已经从最初的“两阶段 warmup + continuation”调整为更直接的 base-model 路线：

- `base_vanilla_N`
- `base_s3ft_N`

其中 `N ∈ {1000, 3000, full}`，用于比较不同训练规模下 vanilla 与 S3FT 的差异。

## 当前结论

- `Vanilla SFT` 在 `GSM8K test` 上可能带来轻微或不稳定的收益。
- `Vanilla SFT` 在 `SVAMP` 上会出现更明显的退化。
- `S3FT` 在当前设置下没有提升 `GSM8K` in-domain accuracy。
- `S3FT` 在 `SVAMP` 上 consistently 比 vanilla 更稳，说明它更像是在缓解继续监督带来的泛化收缩。

主结果见 [results/main_results.md](./results/main_results.md)。

## 主结果

| Model | GSM8K | SVAMP |
| --- | ---: | ---: |
| `base` | 0.37 | 0.76 |
| `base_vanilla_1000` | 0.38 | 0.67 |
| `base_s3ft_1000` | 0.32 | 0.69 |
| `base_vanilla_3000` | 0.34 | 0.66 |
| `base_s3ft_3000` | 0.31 | 0.72 |
| `base_vanilla_full` | 0.36 | 0.68 |
| `base_s3ft_full` | 0.31 | 0.71 |

## 仓库结构

```text
configs/         训练配置
data/            数据与训练集格式
docs/research/   项目计划、执行文档
reports/         实验日志
results/         结果摘要与统计
scripts/         数据处理、生成、训练、评测脚本
```

## 关键脚本

- `scripts/prepare_gsm8k.py`
- `scripts/generate_self_answers.py`
- `scripts/generate_self_answers_vllm.py`
- `scripts/eval_math.py`
- `scripts/build_s3ft_dataset.py`
- `scripts/train_sft_baseline.py`
- `scripts/merge_lora.py`

## 快速复现流程

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成 self-answer

```bash
python scripts/generate_self_answers_vllm.py \
  --input-file data/gsm8k/train.jsonl \
  --output-file data/gsm8k/self_answers_from_base_1000.jsonl \
  --model-path /path/to/Qwen2.5-1.5B-Instruct \
  --max-samples 1000 \
  --temperature 0.7 \
  --top-p 0.9
```

### 3. Judge 并构建 S3FT 数据

```bash
python scripts/eval_math.py \
  --predictions data/gsm8k/self_answers_from_base_1000.jsonl \
  --output-file data/gsm8k/self_answers_from_base_1000_scored.jsonl

python scripts/build_s3ft_dataset.py \
  --scored-input data/gsm8k/self_answers_from_base_1000_scored.jsonl \
  --s3ft-output data/gsm8k/train_s3ft_from_base_1000.jsonl \
  --self-answer-output data/gsm8k/train_self_answer_only_from_base_1000.jsonl \
  --stats-output results/rewrite_stats_from_base_1000.json
```

### 4. 训练

```bash
python scripts/train_sft_baseline.py --config configs/base_vanilla_1000.yaml
python scripts/train_sft_baseline.py --config configs/base_s3ft_1000.yaml
```

### 5. 合并 LoRA

```bash
python scripts/merge_lora.py \
  --base-model-path /path/to/Qwen2.5-1.5B-Instruct \
  --adapter-path outputs/base_vanilla_1000 \
  --output-dir outputs/base_vanilla_1000_merged
```

### 6. 评测

```bash
python scripts/generate_self_answers_vllm.py \
  --input-file data/gsm8k/test.jsonl \
  --output-file results/gsm8k_base_vanilla_1000_preds_det.jsonl \
  --model-path outputs/base_vanilla_1000_merged \
  --temperature 0 \
  --top-p 1.0 \
  --max-new-tokens 512

python scripts/eval_math.py \
  --predictions results/gsm8k_base_vanilla_1000_preds_det.jsonl
```

## 相关文档

- [项目计划书](./docs/research/s3ft_project_plan.md)
- [执行清单](./docs/research/s3ft_execution_checklist.md)
- [实验日志](./reports/experiment_log.md)
- [结果摘要](./results/main_results.md)
