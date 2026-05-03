# 近期后训练 A 会论文梳理（聚焦 SFT / RLHF / Preference Optimization）

更新时间：2026-05-01  
整理范围：优先收录 2024-2025 年已公开为主会/主 track 的 ICLR、NeurIPS、ACL、EMNLP 论文；主题聚焦 `SFT`、`RLHF`、`DPO/Preference Optimization`、`Reward Modeling` 的后训练方法。  
阅读目标：帮你选一个适合做复现并写进简历的项目，不只是“跑通代码”，而是能讲出方法、训练设计、实验逻辑和工程取舍。

## 1. 先说结论：最近大家在干嘛

近两年的后训练论文，主流方向非常清楚，基本集中在下面四条线上：

1. `把 RLHF 简化成更像监督学习的问题`
   代表是 DPO 系方法的继续演化：`SimPO`、`Cal-DPO`、`DiscoPOP`、`BFPO`。核心动机是减少 PPO/RM 管线复杂度，提升稳定性，降低算力和工程门槛。

2. `让偏好监督更丰富，而不是只做 winner vs loser 的二分类`
   代表是 `SPPO`、`TPO`。思路是把偏好看成概率、排序、树结构甚至博弈过程，而不是简单 pairwise binary preference。

3. `开始怀疑“继续优化模型”是不是总比“优化数据”更重要`
   代表是 `Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision`。这是很值得读的一篇，因为它不只是发明 loss，而是挑战现有后训练范式。

4. `SFT 本身被重新审视`
   以前大家把 SFT 当成“预处理阶段”，现在很多论文开始研究：
   `SFT 的数据组合怎么影响能力`、`SFT 会不会破坏多样性`、`是不是能靠 curriculum / phased tuning 做得更好`。

如果你是为了做简历项目，我的判断是：

- 最适合做“算法复现 + 指标复现”的：`SimPO`
- 最适合做“有研究味道的完整项目”的：`SPPO`
- 最适合做“有洞察、有批判性、好讲故事”的：`ILR`
- 最适合做“偏 SFT、低风险、容易做 ablation”的：`GEM` 或 `Phased IFT`

## 2. 候选论文总表

| 论文 | 会议 | 时间 | 主题 | 核心想法 | 复现难度 | 简历价值 |
|---|---|---|---|---|---|---|
| SimPO: Simple Preference Optimization with a Reference-Free Reward | NeurIPS 2024 | 2024-09 | Preference Optimization | 用 `平均 log prob` 当隐式 reward，去掉 reference model，并加入 reward margin | 中 | 很高 |
| Cal-DPO: Calibrated Direct Preference Optimization for Language Model Alignment | NeurIPS 2024 | 2024-12 | Preference Optimization | 给 DPO 的隐式 reward 做 calibration，关注 reward 绝对尺度而不只是相对差值 | 中 | 高 |
| Discovering Preference Optimization Algorithms with and for LLMs (DiscoPOP) | NeurIPS 2024 | 2024-09 | Preference Optimization / Auto-discovery | 用 LLM 自动发现新的 preference loss，找到 logistic + exponential 的混合目标 | 中偏高 | 高 |
| Self-Play Preference Optimization (SPPO) | ICLR 2025 | 2025-01 | RLHF / Preference Optimization | 把偏好优化写成 constant-sum self-play，逼近 Nash equilibrium | 高 | 很高 |
| TPO: Aligning LLMs with Multi-branch & Multi-step Preference Trees | ICLR 2025 | 2025-01 | Reasoning Alignment | 不再只从树里抽 pair，而是直接吃完整 preference tree / ranked list | 高 | 很高 |
| Bi-Factorial Preference Optimization (BFPO) | ICLR 2025 Spotlight | 2025-01 | Safe RLHF / Multi-objective | 把 helpfulness + harmlessness 联合 RLHF 重参数化为单一监督目标 | 中偏高 | 很高 |
| Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning | ICLR 2025 | 2025-01 | Offline RL + SFT | 把 Q-learning 改写成非常接近 SFT 的目标，不新加 value head | 高 | 高 |
| Preserving Diversity in Supervised Fine-Tuning of LLMs (GEM) | ICLR 2025 | 2025-01 | SFT | 用带熵正则的博弈式训练替代普通 CE，减少 over-memorization、保住多样性 | 中 | 很高 |
| Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision (ILR) | ICLR 2025 Spotlight | 2025-01 | Weak Supervision / Post-training | 在监督不可靠时，优先用比较反馈改进数据标签，而不是继续做 DPO | 中 | 很高 |
| Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models | EMNLP 2024 | 2024-11 | SFT + Preference Refinement | 先用强模型蒸馏 reasoning，再让小模型自采样、自 refinement、偏好优化 | 中 | 高 |
| Phased Instruction Fine-Tuning for Large Language Models | Findings ACL 2024 | 2024-08 | SFT Curriculum | 按 instruction 难度分阶段 SFT，验证“逐步对齐”的假设 | 低 | 中高 |
| How Abilities in LLMs are Affected by SFT Data Composition | ACL 2024 Long | 2024-08 | SFT Data Mixture | 系统分析 SFT 数据量、混合比例、能力冲突和遗忘，并提出 DMT | 中 | 高 |

## 3. 我最建议你优先看的 8 篇

下面这 8 篇最适合你拿来做简历项目储备。我把它们分成四类。

### A. 直接做偏好优化算法

#### 3.1 SimPO

一句话：`把 DPO 里最烦的 reference model 拿掉，并让 reward 更贴近生成本身。`

为什么重要：

- DPO 已经是开源后训练的标准 baseline 之一，SimPO 属于“你一定绕不开”的后续升级版。
- 它不是玄学 trick，而是明确改了 reward 形式和 margin 设计。
- 对工程实现很友好，适合在 `TRL / Axolotl / LLaMA-Factory` 这类框架里快速复现。

核心机制：

- 不像 DPO 要比较 `policy` 与 `reference` 的 log-ratio，SimPO 直接用 `response 的平均 log probability` 当隐式 reward。
- 这样一来：
  `不需要 reference model`
  `显存更省`
  `训练逻辑更简单`
- 额外加入 `target reward margin`，逼 preferred response 和 rejected response 拉开间隔。

论文里做了什么：

- 在 `Mistral / Llama 3 / Gemma 2` 等模型上做实验。
- 评价指标覆盖 `AlpacaEval 2`、`MT-Bench`、`Arena-Hard`。
- 论文报告里对 DPO 有明显提升，而且没有明显拉长回答长度。

适合复现的原因：

- 方法清楚，loss 好写。
- 官方代码公开，门槛低。
- 很容易做出有说服力的 ablation：
  `是否去掉 reference model`
  `平均 log prob vs sum log prob`
  `margin 超参敏感性`
  `不同 preference dataset 的泛化`

简历怎么讲：

- “复现并对比 DPO / SimPO 在 UltraFeedback 上的训练稳定性、显存占用与对齐指标”
- “实现 reference-free preference optimization pipeline，并分析长度偏置与 reward margin 的作用”

我的评价：`这是最稳的第一选择。`

#### 3.2 Cal-DPO

一句话：`DPO 只关心 winner 比 loser 大，不关心 reward 尺度本身；Cal-DPO 就是去修这个问题。`

核心洞察：

- 传统 contrastive preference optimization 更关注相对排序，可能忽略 reward 的绝对尺度。
- 这会导致学到的隐式 reward 与真实偏好强度不匹配。

方法重点：

- 给隐式 reward 做 calibration，让 learned reward 和 ground-truth reward 在尺度上更可比。
- 论文强调理论优势，也给了标准 benchmark 上的增益。

适合复现吗：

- 适合。
- 但它的故事性不如 SimPO 强，面试里你需要更清楚地讲“为什么尺度校准值得做”。

更适合谁：

- 想做“我不只是换个 loss，我在修正 DPO 的 reward semantics”的项目。

#### 3.3 DiscoPOP

一句话：`不是人手写 loss，而是让 LLM 帮你发现新的 preference optimization objective。`

为什么值得看：

- 这篇论文不只是后训练，也有 “meta-research / automatic algorithm discovery” 的味道。
- 对面试官来说会显得你关注的不只是训练结果，还有“目标函数搜索”这件事。

方法重点：

- 给 LLM 历史实验结果和性能指标，让它提出新的 loss 形式。
- 迭代搜索后得到 `DiscoPOP`，本质上是对 logistic / exponential 风格目标的自适应混合。

复现难点：

- 如果完整复现“自动发现过程”，成本不低。
- 但如果你只复现最终的 `DiscoPOP loss` 并和 DPO/SimPO 比，很可行。

简历价值：

- 高，但更偏“有趣”和“研究感”，不是最稳的第一枪。

### B. 更强监督结构：从 pairwise 到 self-play / tree

#### 3.4 SPPO

一句话：`把偏好优化从“二分类”提升到“博弈过程”。`

为什么它很强：

- 它不是简单改 loss，而是改了问题建模方式。
- 把语言模型对齐视为常和博弈，目标是逼近 Nash equilibrium policy。
- 用 `PairRM` 这类偏好模型对自博弈样本打分，只用 prompt 就能做出很强效果。

论文亮点：

- 只用 `UltraFeedback` 的 `60k prompts`，不依赖外部更强 LLM 额外生成 responses/preferences。
- 在 `Mistral-7B-Instruct-v0.2`、`Llama-3-8B-Instruct` 上报告了很强结果。
- 对比 `(iterative) DPO / IPO`，在多个 benchmark 上更优。

为什么适合做项目：

- 项目故事完整：`自采样 -> PairRM 打分 -> self-play 更新 -> benchmark 对比`
- 你能写出很漂亮的系统架构图和训练流程图。
- 既有理论，也有工程。

复现难点：

- 比 SimPO 高不少。
- 你需要把 `sampling / ranking / iterative update / evaluation` 都串起来。

简历怎么讲：

- “基于 Self-Play Preference Optimization 复现开源对齐流程，构建无外部教师监督的后训练闭环”
- “分析 PairRM 质量、采样温度与 self-play 迭代次数对对齐效果的影响”

我的评价：`如果你愿意认真做一段时间，这是最像正式研究项目的一篇。`

#### 3.5 TPO

一句话：`如果你已经有 preference tree，就别再暴力采样 pair 了，直接学习整棵树。`

为什么好：

- 它瞄准了 reasoning alignment 的一个真问题：
  复杂推理里，很多 response 不是简单好/坏二元关系，而是有多级偏好和步骤级差异。
- TPO 把任务改成 `Preference List Ranking`，并引入 `Adaptive Step Reward` 做细粒度 credit assignment。

适合什么项目：

- 数学/推理方向的后训练项目。
- 如果你想做 `Tree-of-Thought`、`step-level reward`、`reasoning trace ranking`，这篇很合适。

难点：

- 数据构造复杂。
- 评测也更偏 reasoning，不像通用聊天那样一把梭。

我的评价：`研究味很足，但不适合第一次做后训练项目。`

### C. 重新理解“RLHF 到底在优化什么”

#### 3.6 BFPO

一句话：`把 helpfulness 和 harmlessness 的联合 RLHF，改写成一个监督学习目标。`

为什么值得关注：

- 安全对齐里常见问题不是单目标优化，而是多目标冲突。
- BFPO 很适合做“安全 + 有用性权衡”的简历项目，话题性比普通 DPO 更强。

关键点：

- 不是简单加权两个 reward，而是通过 labeling function 去编码 global preference ranking。
- 目标是同时把 `helpfulness` 和 `harmlessness` 纳入一个统一监督优化。
- 论文声称能在显著更低的计算资源下达到同级别安全效果。

适合谁：

- 想把项目做成“对齐/安全方向”的人。
- 简历目标偏研究岗、对齐岗、安全岗。

注意：

- 评测面会更宽，最好补上 `toxicity / refusal / harmlessness-helpfulness tradeoff` 这类分析。

#### 3.7 ILR：Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision

一句话：`监督不可靠时，别急着继续 DPO，先把训练数据修好。`

这是我非常推荐读的一篇，因为它在挑战一个默认前提：

- 传统后训练流程是 `SFT -> preference comparisons -> DPO/RLHF`
- 这篇说：当人类演示和比较反馈都不太可靠时，`SFT + DPO` 会失效，尤其 DPO 很难继续提升
- 更好的做法是用比较反馈去决定：
  `原来的 demonstration 要不要被模型生成的新答案替换`
  然后重新 SFT

为什么这很有价值：

- 这是“数据中心主义”的后训练思路，不再迷信 loss engineering。
- 对现实世界很合理，尤其是复杂数学、代码、安全场景，人类监督经常不稳定。

项目价值非常高：

- 你可以做成一个特别漂亮的故事：
  `弱监督环境下，数据 refinement 比 preference optimization 更重要`
- 很适合写成博客或技术总结，面试时会显得你有判断力，而不是只会照着论文换 loss。

我的评价：`这是最适合做“有观点的简历项目”的候选。`

### D. SFT 自身的升级路线

#### 3.8 GEM：Preserving Diversity in Supervised Fine-Tuning of LLMs

一句话：`SFT 不该只会背答案，还要保住生成多样性。`

为什么这篇值得做：

- 它不依赖复杂 preference data，项目风险更低。
- 同时它讨论的是一个非常真实的问题：`CE SFT 会把模型越训越“死”`，多样性下降、遗忘增加。

方法核心：

- 用博弈式 formulation + entropy regularization 替换单纯 CE。
- 保持输出多样性，同时减轻 over-memorization 和 forgetting。

为什么简历里好讲：

- 很容易定义对照实验：
  `CE vs GEM`
  `pass@k / self-consistency / diversity metrics`
  `chat / code / reasoning` 上的变化
- 官方代码公开，落地门槛好很多。

我的评价：`如果你想做一个偏 SFT、风险小、但看起来不浅的项目，GEM 很合适。`

## 4. ACL / EMNLP 里值得补充看的几篇

这几篇未必是你最终第一复现优先级，但非常适合补背景，帮助你形成方法判断。

### 4.1 Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models

核心思路：

- 第一阶段先用强模型合成 demonstration，把 reasoning transfer 给小模型。
- 第二阶段让小模型自己采样多条 reasoning path，再做 refinement / preference-style optimization。

价值：

- 这是 `SFT + self-improvement + preference optimization` 的组合式范式。
- 对做 reasoning alignment 很有启发。

### 4.2 Phased Instruction Fine-Tuning

核心思路：

- 按 instruction 难度从低到高分阶段训练，而不是把所有数据一次性混进去。

价值：

- 实现简单。
- 特别适合作为“小而美”的复现项目或 baseline 增强模块。

### 4.3 How Abilities in LLMs are Affected by SFT Data Composition

核心思路：

- 系统分析 SFT 数据量、能力类型、数据混合比例和遗忘。
- 论文发现不同能力对数据规模的 scaling pattern 不一样，而且多技能混训会出现冲突。

价值：

- 如果你最后真要自己做项目，这篇能帮你少踩很多坑。
- 它告诉你：不是“多混一点数据”就一定更好。

## 5. 如果从“适合复现做简历项目”来排序

### 第一梯队：最推荐

#### 1) SimPO

为什么第一：

- 经典、主流、清晰、工程友好。
- 做出来以后，任何人一看都知道你在复现真正的后训练方法。
- 能讲 loss、显存、reference-free 设计、训练稳定性、评测。

适合项目标题：

- `基于 SimPO 的开源 LLM 偏好对齐复现与扩展`
- `Reference-free Preference Optimization for LLM Alignment`

#### 2) SPPO

为什么第二：

- 研究味道最强，完整度很高。
- 有训练闭环，不只是换 loss。

适合项目标题：

- `基于 Self-Play Preference Optimization 的无教师后训练对齐系统`

#### 3) ILR

为什么第三：

- 观点新，讨论的是弱监督现实问题。
- 很适合做成“方法复现 + 我自己的分析”的项目。

适合项目标题：

- `弱监督后训练中的数据修正优于偏好优化：ILR 复现与分析`

### 第二梯队：很不错，但更看方向匹配

- `GEM`
  适合偏 SFT、想降低项目风险的人。
- `BFPO`
  适合偏安全对齐的人。
- `TPO`
  适合偏推理、多步轨迹、tree preference 的人。
- `Cal-DPO`
  适合偏理论解释和 reward semantics 的人。

## 6. 我建议你怎么选

如果你目标是 `两到四周做出一个像样的简历项目`，我建议按下面规则选：

1. `想稳妥出成果`
   选 `SimPO`

2. `想做出最有研究味、最完整的项目`
   选 `SPPO`

3. `想做出有思考、有观点的项目`
   选 `ILR`

4. `你手头算力一般，想先做 SFT 方向`
   选 `GEM` 或 `Phased IFT`

## 7. 每个候选项目的大致工作量

### SimPO

- 最低可行版本：`3-5 天`
- 完整项目版本：`1-2 周`
- 需要内容：
  数据清洗、DPO baseline、SimPO loss、训练日志、AlpacaEval/MT-Bench 或轻量替代评测、ablation

### SPPO

- 最低可行版本：`1-2 周`
- 完整项目版本：`2-4 周`
- 需要内容：
  prompt-only 数据、采样、PairRM 或替代 ranker、self-play 迭代、评测闭环

### ILR

- 最低可行版本：`1 周`
- 完整项目版本：`2-3 周`
- 需要内容：
  构造弱监督 setting、SFT baseline、DPO baseline、label refinement pipeline、数据质量分析

### GEM

- 最低可行版本：`4-7 天`
- 完整项目版本：`1-2 周`
- 需要内容：
  CE baseline、GEM loss、diversity 指标、forgetting 分析、test-time sampling 对比

## 8. 如果是我替你做选择

我会给出两个非常实际的路线：

### 路线 A：稳、好讲、最适合第一段简历项目

选 `SimPO`

项目结构：

1. 基于现成 SFT 基座模型做 DPO/SimPO 对比
2. 用 UltraFeedback 或 HH-RLHF 做 preference tuning
3. 比较：
   训练稳定性
   显存占用
   win-rate / judge score
   输出长度偏置
4. 加 2-3 个 ablation，形成完整故事

这个路线最容易做出“像论文复现”的质感。

### 路线 B：更有深度，面试更容易聊出内容

选 `ILR`

项目结构：

1. 人为构造 weak supervision 场景
2. 复现 `SFT -> DPO` baseline
3. 实现比较反馈驱动的 label refinement
4. 分析不同噪声水平下：
   `SFT`
   `SFT + DPO`
   `SFT + ILR`
5. 给出结论：什么时候应该“修模型”，什么时候应该“修数据”

这个路线更有你的个人判断，不容易沦为“照抄论文”。

## 9. 可直接参考的公开代码线索

- SimPO 官方代码：<https://github.com/princeton-nlp/SimPO>
- SPPO 官方代码：<https://github.com/uclaml/SPPO>
- GEM 官方代码：<https://github.com/liziniu/GEM>
- Cal-DPO 官方代码：<https://github.com/tengxiao1/Cal-DPO>
- ILR 代码与数据：<https://github.com/helloelwin/iterative-label-refinement>
- BFPO 训练配方：<https://github.com/wx-zhang/bfpo>

## 10. 参考论文链接

### NeurIPS / ICLR 主线

- SimPO: Simple Preference Optimization with a Reference-Free Reward  
  <https://openreview.net/forum?id=3Tzcot1LKb>

- Cal-DPO: Calibrated Direct Preference Optimization for Language Model Alignment  
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/cf8b2205e39f81726a8d828ecbe00ad0-Abstract-Conference.html>

- Discovering Preference Optimization Algorithms with and for Large Language Models  
  <https://openreview.net/forum?id=erjQDJ0z9L>

- Self-Play Preference Optimization for Language Model Alignment  
  <https://openreview.net/forum?id=a3PmRgAB5T>

- TPO: Aligning Large Language Models with Multi-branch & Multi-step Preference Trees  
  <https://openreview.net/forum?id=O0sQ9CPzai>

- Bi-Factorial Preference Optimization: Balancing Safety-Helpfulness in Language Models  
  <https://openreview.net/forum?id=GjM61KRiTG>

- Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning  
  <https://openreview.net/forum?id=v4MTnPiYXY>

- Preserving Diversity in Supervised Fine-Tuning of Large Language Models  
  <https://openreview.net/forum?id=NQEe7B7bSw>

- Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision  
  <https://openreview.net/forum?id=q5EZ7gKcnW>

### ACL / EMNLP 补充

- Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models  
  <https://aclanthology.org/2024.emnlp-main.139/>

- Phased Instruction Fine-Tuning for Large Language Models  
  <https://aclanthology.org/2024.findings-acl.341/>

- How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition  
  <https://aclanthology.org/2024.acl-long.12/>

## 11. 最后给你的一个建议

不要把项目目标定成“我把论文跑通”。更好的目标是：

`我复现一个后训练方法 + 做出一组有结论的对比实验 + 能说清楚它为什么有效、什么时候失效、工程上值不值得。`

如果最后你想做成真正能写进简历、甚至面试里能聊 10 分钟以上的项目，我建议优先从下面三篇里选：

1. `SimPO`
2. `SPPO`
3. `ILR`

其中：

- `SimPO` 最稳
- `SPPO` 最完整
- `ILR` 最有观点

