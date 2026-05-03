# S3FT 扩展版项目计划书

更新时间：2026-05-03  
项目主题：`Selective Self-to-Supervised Fine-Tuning for Generalization in Large Language Models` 的复现与扩展  
目标：把一个“方法本身不重”的 SFT 论文，做成一个有分析深度、能写进简历、能在面试里讲 10 分钟的完整项目。

## 1. 项目一句话定义

这个项目不只是复现 `S3FT`，而是要回答一个更大的问题：

`标准 SFT 是否因为过度拟合单一 gold answer 而损害模型泛化？如果选择性引入模型自生成的正确答案，能否在保持任务性能的同时保住更强的通用能力与后续可塑性？`

这句话很重要，因为它决定了项目不是“小修小补的 SFT trick”，而是：

`一个关于后训练监督形式如何影响泛化与能力保留的项目。`

## 2. 项目背景

标准 SFT 的基本假设是：

- 每个 prompt 对应一个最值得学习的 reference answer
- 模型应该尽量贴近这个 answer

但现实里常常不是这样：

- 很多任务有多个正确答案
- gold answer 只是许多可接受答案中的一个
- 强行逼模型只学这一个答案，容易让模型分布变窄
- 分布一窄，模型在 out-of-domain 泛化、best-of-n、self-consistency、后续 preference tuning 上可能都吃亏

S3FT 的基本想法是：

- 先让模型对训练样本自己回答
- 用 judge 判断模型答案是否正确
- 如果正确，就让模型学习自己的答案，而不是强行只学 gold
- 如果不正确，再退回 gold 或 gold paraphrase

它本质上在做一件事：

`把监督信号从“唯一标准答案”放宽为“多种可接受答案中的一个”，从而减少过拟合。`

## 3. 为什么值得做成项目

S3FT 单看方法，确实容易显得轻。但它的潜力在于：

1. `问题真实`
   很多后训练和 instruction tuning 的核心问题，本质上就是过度拟合单一模板答案。

2. `和 2025 趋势一致`
   2025 年后训练越来越重视：
   数据质量、泛化、多样性、保留预训练能力，而不只是把 target task 分数拉满。

3. `容易做出自己的分析`
   你可以从：
   judge 质量
   自生成答案比例
   不同任务的多解空间
   和后续 DPO / test-time scaling 的兼容性
   这些角度做出比论文更有项目感的内容。

4. `资源压力相对可控`
   它不像 RLVR 或 SPPO 那样重，个人更容易在有限算力下做完整闭环。

## 4. 项目目标

这个项目建议分成三个层级目标。

### 目标 A：基础复现

复现 S3FT 的核心流程，并在一个或两个任务上验证：

- `S3FT` 是否能优于 `Vanilla SFT`
- 是否能更好保留泛化能力

### 目标 B：扩展分析

系统研究下面三个问题：

1. `S3FT 的收益到底来自 selective 机制，还是来自“答案更多样”本身`
2. `judge 的质量对最终结果影响多大`
3. `S3FT 是否能改善后续 test-time scaling 或 post-training compatibility`

### 目标 C：简历级结论

最后你要能得出 2 到 3 条很清晰的结论，例如：

- 在多解任务上，`S3FT` 比标准 `SFT` 更能保持 out-of-domain 泛化
- `judge precision` 比 `judge recall` 更重要
- 引入模型自生成正确答案，有助于提升 `best-of-n` 或后续 `DPO cold start`

这类结论才是真正能写进简历、博客和面试回答里的东西。

## 5. 核心研究问题

这个项目最值得围绕下面 5 个问题设计。

### Q1. S3FT 相比 Vanilla SFT，提升的到底是什么

要区分：

- target task accuracy 提升
- generalization 保留更好
- 输出多样性更强
- 后续可塑性更高

### Q2. S3FT 的效果是不是依赖任务类型

可以比较：

- 数学推理
- 代码生成
- instruction following
- 开放式 QA

我的预判是：

- 多解空间大的任务，S3FT 更有效
- 单一标准答案强约束任务，收益可能有限

### Q3. Judge 的质量会不会成为瓶颈

这是最值得扩展的点之一。

你需要回答：

- 如果 judge 很强，S3FT 是否稳定有效
- 如果 judge 有噪声，会不会把错误答案带进训练
- precision / recall 哪个更关键

### Q4. 模型自己的答案为什么可能比 gold 更值得学

这个问题是项目的思想核心。你可以从下面几方面分析：

- self answer 是否更贴近模型原本分布
- self answer 是否减少了 train-test mismatch
- self answer 是否帮助保留预训练阶段形成的表达空间

### Q5. S3FT 是否能作为更好的后续起点

这是把项目做厚的关键：

- S3FT 训练后的模型，是否更适合继续做 DPO
- 是否更适合 best-of-n / self-consistency
- 是否更不容易在第二阶段后训练中塌缩

## 6. 项目方案总览

推荐做成三阶段。

### 阶段 1：主复现

建立四条 baseline：

1. `Vanilla SFT`
2. `S3FT`
3. `Self-answer only`
4. `Gold + paraphrase`

其中：

- `Vanilla SFT` 负责做标准基线
- `S3FT` 是主方法
- `Self-answer only` 用来验证是否必须“selective”
- `Gold + paraphrase` 用来验证收益是不是只是因为答案表面变多样了

### 阶段 2：Judge 研究

至少做两类 judge：

1. `Strong judge`
   例如 GPT-4 级别 judge 或更强 verifier

2. `Weak judge`
   例如 rule-based、较小本地模型、或者自动评分器

比较：

- 不同 judge 对训练集重写比例的影响
- 不同 judge 对最终性能的影响
- false positive / false negative 哪类更伤

### 阶段 3：泛化与后续可塑性

做两种扩展验证：

1. `Generalization`
   比较 in-domain 和 out-of-domain 表现

2. `Downstream compatibility`
   在 SFT 后接一个轻量后续阶段，例如：
   best-of-n reranking
   self-consistency
   或小规模 DPO

这一步会让你的项目明显比普通复现厚很多。

## 7. 实验设计

## 7.1 数据集建议

建议不要一开始就铺太大，先做一个主任务，再加一个补充任务。

### 方案 A：数学推理主线

训练：

- `GSM8K`
- 可选 `MetaMathQA` 子集

评测：

- `GSM8K`
- `MATH` subset
- `SVAMP`

优点：

- judge 相对容易做
- answer correctness 可自动判定
- 泛化分析清楚

### 方案 B：代码生成主线

训练：

- `MBPP`
- `HumanEval` 风格 instruction data 或其训练替代集

评测：

- `MBPP`
- `HumanEval`
- pass@k

优点：

- verifier 很自然
- 很适合分析 “多正确答案” 问题

### 方案 C：通用 instruction tuning 主线

训练：

- `Alpaca`
- `UltraChat` 子集
- `OpenHermes` 子集

评测：

- instruction-following
- MT-Bench judge
- AlpacaEval 风格自动评测

优点：

- 更贴近通用对话场景

缺点：

- judge 质量会成为大问题

如果是个人项目，我更推荐：

`先做数学或代码。`

因为 judge 更可靠，项目结论会更硬。

## 7.2 模型建议

根据资源给三个档位。

### 低资源

- `Qwen2.5-1.5B`
- `Llama-3.2-1B`
- `Phi-3-mini`

适合：

- 跑通全流程
- 快速做 ablation

### 中资源

- `Qwen2.5-3B`
- `Llama-3.2-3B`

适合：

- 做正式项目主结果

### 高资源

- `Qwen2.5-7B`
- `Llama-3-8B`

适合：

- 做更强结果和更像论文的展示

建议策略：

`先用 1.5B/3B 跑出结论，再决定要不要上 7B。`

## 7.3 Judge 设计

这是项目成败关键。

建议做三档 judge。

### Judge A：Exact verifier

适用：

- 数学最终答案
- 代码单元测试

优点：

- 最可靠

### Judge B：LLM-as-a-judge

适用：

- 开放问答
- 多步推理解释质量

优点：

- 灵活

缺点：

- 成本高
- 有噪声

### Judge C：Weak heuristic judge

适用：

- 做鲁棒性对照

例子：

- exact match
- execution pass/fail 的弱化版本
- 小模型 judge

### 你至少要做的分析

1. 重写比例
2. judge 一致性
3. false positive / false negative 的误差类型
4. 最终性能对 judge 质量的敏感度

## 7.4 评价指标

不能只看一个最终分数。

建议指标分 4 组。

### 任务性能

- accuracy
- pass@1
- pass@k

### 泛化性能

- out-of-domain accuracy
- cross-dataset transfer

### 分布/多样性相关

- self-consistency gain
- best-of-n gain
- 不同采样温度下的稳定性

### 训练数据重写分析

- 被替换样本比例
- 按难度分桶后的替换比例
- self-answer 与 gold 的 lexical overlap
- self-answer 长度分布

## 8. 推荐的扩展实验

这是让项目从“复现”变成“有内容”的关键部分。

## 实验 1：Selective 是否真有必要

比较：

- `Vanilla SFT`
- `Self-answer only`
- `S3FT`

要回答：

`收益是不是来自“模型自己的正确答案”，还是必须靠 selective 机制控制噪声。`

## 实验 2：Judge 质量敏感性

比较：

- 强 judge
- 弱 judge
- 人工抽样校验

要回答：

`S3FT 到底有多依赖 judge 质量。`

## 实验 3：多解空间大小的影响

按任务类型或样本特征分组：

- 单一答案型
- 多解表达型

要回答：

`S3FT 是否主要在“存在多个合理答案”的任务中有效。`

## 实验 4：对后续 test-time scaling 的影响

比较：

- `CE-SFT` 和 `S3FT`
- 在 `best-of-n` 或 `self-consistency` 下的差距

要回答：

`S3FT 是不是让模型保留了更可利用的答案分布。`

## 实验 5：对后续 preference tuning 的影响

做一个小规模二阶段实验：

- `CE-SFT -> DPO`
- `S3FT -> DPO`

要回答：

`S3FT 是否更适合作为后续对齐的初始化。`

这个实验非常加分，即使规模不大也值得做。

## 9. 项目实施时间线

按 3 周到 5 周设计比较合理。

### 第 1 周：搭框架

- 选任务和模型
- 跑通 Vanilla SFT baseline
- 跑通 self-answer 生成
- 实现 judge 与样本重写

交付物：

- baseline 结果
- S3FT 数据构造脚本

### 第 2 周：主实验

- 跑 `S3FT`
- 跑 `Self-answer only`
- 跑 `Gold + paraphrase`
- 记录主要结果

交付物：

- 主表格
- 重写比例分析

### 第 3 周：关键扩展

- 做 judge 强弱对比
- 做 in-domain / out-of-domain 对比
- 做 self-consistency 或 best-of-n 分析

交付物：

- 核心 ablation
- 结论初稿

### 第 4-5 周：进阶增强

- 做 `S3FT -> DPO` 小规模实验
- 清理图表
- 写技术报告 / 博客 / README

交付物：

- 完整项目文档
- 简历可写版本

## 10. 项目风险与对策

### 风险 1：S3FT 提升不明显

对策：

- 选多解空间更大的任务
- 不只看 accuracy，要看 generalization 与 best-of-n

### 风险 2：Judge 噪声太大

对策：

- 先从可验证任务入手
- 用强 verifier 做主结果
- 再引入弱 judge 做扩展

### 风险 3：项目显得太轻

对策：

- 必须补：
  `judge analysis`
  `generalization`
  `downstream compatibility`

### 风险 4：算力不够

对策：

- 先用 1.5B 或 3B
- 先做数学或代码单任务
- 重点保证结论，不强求大模型主结果

## 11. 最后应该产出什么

至少要有下面这些可见成果：

1. 一个可复现的 `S3FT` 训练与数据重写 pipeline
2. 一组 `SFT vs S3FT` 的主实验结果
3. 一组 judge 相关 ablation
4. 一组 generalization 或 best-of-n 分析
5. 一份技术报告或博客

## 12. 简历上怎么写

不要写成：

`复现了 S3FT 论文。`

这种写法太弱。

建议写成：

`设计并实现选择性自监督微调框架，研究单一 gold-answer 监督对大模型泛化的影响；在数学/代码任务上系统比较 Vanilla SFT、S3FT 与多种 judge 策略，分析其对 out-of-domain 泛化、best-of-n 与后续对齐可塑性的影响。`

如果做了二阶段实验，还可以加一句：

`验证了 S3FT 作为后续 preference tuning 冷启动初始化的潜力。`

## 13. 我对这个项目的最终判断

如果你愿意把它按上面的方式做，S3FT 就不再是一个“小 judge 技巧”，而会变成一个很像样的项目：

- 它不靠重 RL
- 它不靠炫公式
- 它靠的是：
  `把监督形式、泛化能力、答案分布和后续可塑性连成一条完整问题链`

这类项目在面试里通常很好讲，因为它既有工程实现，也有方法判断，还有实验设计能力。

