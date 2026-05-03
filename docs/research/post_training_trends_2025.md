# 2025 年后训练趋势图（不只看 RLHF）

更新时间：2026-05-01  
观察窗口：2025 年公开论文，优先参考 `ICLR 2025`、`NAACL 2025`、`ACL 2025`、`EMNLP 2025`、`NeurIPS 2025`。  
关注对象：大语言模型后训练，包括 `SFT`、`Preference Optimization`、`RLHF`、`RLVR`、`推理后训练`、`推理时对齐`、`验证器/奖励设计`。

## 1. 一句话结论

2025 年后训练最明显的变化不是“RL 没人做了”，而是：

`后训练正在从“单一路径的 PPO-RLHF”分叉成多个更实用的方向：离线偏好优化、数据自生成与自修正、推理时对齐、基于可验证奖励的 RL、以及更关注多样性和泛化的 SFT。`

换句话说，2025 年的研究重点已经从：

- `怎么把 reward model + PPO 跑起来`

转向：

- `怎么用更低成本做对齐`
- `怎么让监督信号更可靠`
- `怎么让推理能力真正提升，而不只是表面分数变高`
- `怎么把训练阶段的一部分工作挪到 inference time`

## 2. 2025 的六条主线

## 趋势一：经典 PPO-RLHF 不再是中心，Preference Optimization 成为默认主舞台

2025 年最稳定的一条主线仍然是 `DPO 之后怎么办`。

大家没有放弃“用偏好数据做对齐”，但越来越少人愿意回到完整的 `reward model + PPO` 管线。原因很现实：

- PPO-RLHF 工程复杂
- 训练不稳定
- reward hacking 风险高
- 开源复现成本大

于是 2025 的很多论文都在改写 preference optimization 的目标、数据结构或训练流程。

代表论文：

- `SPPO`，ICLR 2025  
  <https://openreview.net/forum?id=a3PmRgAB5T>

- `InfoPO`，NAACL 2025  
  <https://aclanthology.org/2025.naacl-long.585/>

- `TPO`，ICLR 2025  
  <https://openreview.net/forum?id=O0sQ9CPzai>

- `SimPER`，ICLR 2025  
  <https://openreview.net/forum?id=jfwe9qNqRi>

- `Binary Classifier Optimization`，ACL 2025  
  <https://aclanthology.org/2025.acl-long.93/>

你可以把这条趋势理解成：

`2025 年很多人仍在做“偏好对齐”，但更像在做离线目标设计，而不是做传统 RLHF。`

## 趋势二：数据比 loss 更重要，自动化偏好数据生成与标签修正变成热点

2025 年一个很强的信号是，很多论文不再默认“只要 loss 足够好，模型自然会对齐”。大家开始越来越认真地问：

- preference data 从哪来
- 数据噪声有多大
- 弱监督下该继续训模型，还是先修数据

这导致两类工作变热：

1. `自动生成 preference data`
2. `用比较反馈反过来修 demonstration / label`

代表论文：

- `Self-Steering Optimization (SSO)`，ACL Findings 2025  
  <https://aclanthology.org/2025.findings-acl.473/>

- `Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision`，ICLR 2025 Spotlight  
  <https://openreview.net/forum?id=q5EZ7gKcnW>

- `Aligning LLMs with Implicit Preferences from User-Generated Content`，ACL 2025  
  <https://aclanthology.org/2025.acl-long.384/>

- `ROPO`，ICLR 2025 submission track  
  <https://openreview.net/forum?id=0nxocR2qx4>

这条趋势背后的判断很重要：

`2025 年后训练越来越数据中心化。`

研究者开始承认，复杂任务里人类偏好监督本来就不稳定，先把数据质量、标签可靠性、反馈来源搞对，往往比继续堆更复杂的优化器更值。

## 趋势三：SFT 被重新定义，不再只是“第一阶段预处理”

2025 年还有一条容易被忽略、但其实很关键的趋势：

`SFT 本身重新变成研究对象。`

过去很多工作把 SFT 当成固定前置步骤；2025 年不少论文开始研究：

- SFT 会不会过拟合某种回答风格
- SFT 会不会损伤多样性
- SFT 如何在小数据下保持泛化
- SFT 如何给后续 RL 或 reasoning 留出探索空间

代表论文：

- `Preserving Diversity in Supervised Fine-Tuning of Large Language Models`，ICLR 2025  
  <https://openreview.net/forum?id=NQEe7B7bSw>

- `Selective Self-to-Supervised Fine-Tuning (S3FT)`，NAACL Findings 2025  
  <https://aclanthology.org/2025.findings-naacl.349/>

- `Semi-supervised Fine-tuning for Large Language Models`，NAACL Findings 2025  
  <https://aclanthology.org/2025.findings-naacl.151/>

- `HFT: Half Fine-Tuning for Large Language Models`，ACL 2025  
  <https://aclanthology.org/2025.acl-long.626/>

这说明一个很现实的共识已经出现：

`如果 SFT 把模型训“窄”了，后面很多对齐和 RL 提升空间会一起变小。`

所以 2025 的 SFT 研究不再只是追求 instruction-following 分数，而是开始管：

- diversity
- entropy
- forgetting
- generalization
- 后续 test-time scaling / RL 的可塑性

## 趋势四：对齐开始往 inference time 迁移

2025 年不少工作在尝试把一部分“训练时做的对齐”迁到“推理时做”。

这条线背后的逻辑是：

- 训练代价越来越高
- 用户偏好越来越多样，静态训练难一次覆盖
- 很多对齐问题本质上是 search / reranking / rerouting 问题

代表论文：

- `DiffPO: Diffusion-styled Preference Optimization for Inference Time Alignment`，ACL 2025  
  <https://aclanthology.org/2025.acl-long.926/>

- `MetaAlign: Align LLMs with Diverse Preferences during Inference Time`，NAACL Findings 2025  
  <https://aclanthology.org/2025.findings-naacl.324/>

- `Reward-Guided Tree Search for Inference Time Alignment of LLMs`，NAACL 2025  
  <https://aclanthology.org/2025.naacl-long.625/>

这条线说明 2025 的一个新想法：

`alignment 不一定都要靠继续更新参数，也可以靠 inference-time control、search、reranking、policy-agnostic modules 来做。`

如果你以后做项目，这一方向特别适合算力有限的情况，因为你可以：

- 少训甚至不训大模型
- 把重点放在 reranker / verifier / tree search / sentence-level alignment 上

## 趋势五：推理后训练进入 RLVR 时代，验证器和可验证奖励变成核心

到了 2025 年下半年，最强的一条增量趋势是：

`RL 从 RLHF 向 RLVR 迁移。`

这里的 RLVR 指的是 `Reinforcement Learning with Verifiable Rewards`。  
它在数学、代码、逻辑谜题、程序推导等任务上特别强，因为 reward 可以自动判定。

代表论文：

- `RISE: Trust, But Verify`，NeurIPS 2025  
  <https://openreview.net/forum?id=gA3fFAEXNT>

- `ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in LLMs`，NeurIPS 2025  
  <https://openreview.net/forum?id=YPsJha5HXQ>

- `RL Tango: Reinforcing Generator and Verifier Together`，NeurIPS 2025  
  <https://openreview.net/forum?id=JRkFZl0TJ2>

- `Beyond Verifiable Rewards: Scaling RL to Unverifiable Data`，NeurIPS 2025  
  <https://openreview.net/forum?id=pc6M9h3T9m>

- `Enigmata`，NeurIPS 2025 Spotlight  
  <https://openreview.net/forum?id=fmnxunacr4>

更有意思的是，2025 年这条线内部已经开始“自我反思”了：

- `Does Reinforcement Learning Really Incentivize Reasoning Capacity Beyond the Base Model?`，NeurIPS 2025 Oral  
  <https://openreview.net/forum?id=4OsgYD7em5>

这篇的意义很大，因为它不是继续报更高分，而是在问：

`RLVR 到底是真的让模型学会了新推理，还是只是把底模里本来潜在的高分轨迹筛出来了？`

这标志着 2025 年后训练已经从“会不会用 RL”进化到“RL 到底改变了什么”。

## 趋势六：研究开始更关心“能力边界”和“机制解释”，不只关心 benchmark 提升

2025 年还有一个很重要但不那么显眼的变化：

`论文越来越喜欢问机制问题，而不是只问 final score。`

典型问题包括：

- 偏好优化到底在学什么
- RL 是在创造新能力，还是只在做搜索放大
- 多目标对齐如何平衡 helpfulness 和 harmlessness
- 长链推理中的奖励到底应该给 outcome 还是过程

代表论文：

- `Bi-Factorial Preference Optimization (BFPO)`，ICLR 2025 Spotlight  
  <https://openreview.net/forum?id=GjM61KRiTG>

- `Value-Incentivized Preference Optimization`，ICLR 2025  
  <https://openreview.net/forum?id=SQnitDuow6>

- `Preference Optimization for Reasoning with Pseudo Feedback`，ICLR 2025 Spotlight  
  <https://openreview.net/forum?id=jkUp3lybXf>

- `Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?`，NeurIPS 2025 Oral  
  <https://openreview.net/forum?id=4OsgYD7em5>

这说明后训练领域正在成熟：

`大家不再满足于“这个 loss 赢了”，而开始追问它为什么赢、在哪些监督条件下会失效、它到底改变了模型分布的哪一部分。`

## 3. 2025 趋势的总体判断

如果把 2025 年后训练研究压缩成一句更学术一点的话，可以这么说：

`后训练正在从“基于人类偏好的单阶段参数更新”演化为“训练时优化 + 数据修正 + 推理时控制 + 可验证奖励 + 验证器协同”的组合系统。`

也就是说，2025 年你很少再看到一个项目只靠下面这条线讲完整故事：

`SFT -> reward model -> PPO`

更常见的新叙事已经变成：

- `SFT 先保住多样性和泛化`
- `偏好数据尽量自动生成或自动清洗`
- `偏好优化尽量离线、简单、可扩展`
- `复杂推理尽量交给 verifier 或可验证奖励`
- `必要时把 alignment 放到 inference time 做`

## 4. 对你选项目有什么启发

如果你的目标是选一个 `2025 风格` 的简历项目，我建议按下面的方式理解机会。

### 最像 2025 主流的项目方向

1. `自动化偏好数据 + preference optimization`
   代表：`SSO`、`ILR`

2. `推理对齐 + verifier / RLVR`
   代表：`RISE`、`RL Tango`、`Enigmata`

3. `保留多样性的 SFT / 可泛化 SFT`
   代表：`GEM`、`S3FT`、`HFT`

4. `inference-time alignment`
   代表：`DiffPO`、`MetaAlign`、`Reward-Guided Tree Search`

### 如果只从“可复现性 + 简历效果”来挑

- 最稳：`GEM` 或 `ILR`
- 最有研究味：`SPPO` 或 `RISE`
- 最适合算力有限：`DiffPO` 或 `MetaAlign`
- 最适合做 reasoning 项目：`RL Tango` 或 `Enigmata`

## 5. 如果你只记住三句话

1. `2025 不是不做 RL，而是少做传统 PPO-RLHF，多做 preference optimization 和 RLVR。`
2. `2025 的后训练越来越数据中心化、验证器中心化、推理时控制化。`
3. `真正新的竞争点，已经从“谁的 loss 更花”转向“谁能用更可靠的监督信号，把模型能力稳定地推出来”。`

