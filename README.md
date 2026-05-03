# S3FT 泛化复现项目

这是一个围绕论文 `Selective Self-to-Supervised Fine-Tuning for Generalization in Large Language Models` 的复现与扩展项目。

项目想回答的核心问题是：

`标准 SFT 会不会因为过度拟合单一 gold answer，导致模型泛化能力下降？如果选择性地使用模型自己生成的正确答案进行训练，能否更好地保留泛化能力和后续可塑性？`

## 当前范围

第一阶段先聚焦数学推理场景：

- 任务：`GSM8K -> GSM8K / SVAMP / MATH subset`
- 模型：`Qwen2.5-1.5B` 或 `Qwen2.5-3B`
- 对比方法：
  - `Vanilla SFT`
  - `S3FT`
  - `Self-answer only`
- Judge：基于数值归一化的精确答案匹配

## 仓库结构

```text
configs/         训练配置
data/            原始数据与重写后的训练集
docs/research/   与 S3FT 直接相关的项目计划和执行文档
reports/         实验日志与案例分析
results/         指标、表格和导出的结果
scripts/         数据处理、生成、训练、评测脚本
```

## 计划流程

1. 整理 `GSM8K` 数据为统一指令格式
2. 训练 `Vanilla SFT` baseline
3. 在训练集上生成模型自己的回答
4. 用 judge 判断回答是否正确
5. 构建 `S3FT` 重写训练集
6. 重新训练并和 baseline 对比
7. 分析泛化、judge 质量和 test-time scaling 效果

## 第一阶段目标

- 跑通可复现的 `Vanilla SFT` 配置
- 实现 self-answer 生成脚本
- 实现数学任务的数值 judge
- 实现 `S3FT` 数据集构建脚本
- 统一评测 `GSM8K`、`SVAMP` 和 `MATH subset`

## 项目文档

- [S3FT 项目计划书](./docs/research/s3ft_project_plan.md)
- [S3FT 可执行任务清单](./docs/research/s3ft_execution_checklist.md)

## 环境安装

建议创建独立 Python 环境后安装依赖：

```bash
pip install -r requirements.txt
```

## 当前状态

项目骨架已初始化，接下来会优先补齐数学推理版本的最小可跑闭环。
