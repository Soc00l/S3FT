# ILR / SPPO / RISE / S3FT 项目评估

更新时间：2026-05-02  
目标：不是单纯介绍论文，而是帮助快速判断这 4 个题目 `值不值得做`、`适不适合做简历项目`、`复现难不难`。  
评估维度：`思路`、`创新点`、`项目优点`、`项目短板`、`可行性`、`适合什么人做`。

## 1. 总览结论

如果先只给一句判断：

- `ILR`：最有观点，最适合做“我不只是复现，我有判断”的项目
- `SPPO`：最像正式研究项目，完整度最高，但工程量最大
- `RISE`：最贴近 2025 下半年新趋势，适合做 reasoning + RLVR 方向
- `S3FT`：最稳、最轻、最容易落地，但研究冲击力弱于前 3 个

如果你是为了做一个能写进简历、还能在面试里讲明白的项目，我会这样排：

1. `ILR`
2. `SPPO`
3. `RISE`
4. `S3FT`

但如果你现在算力有限、时间有限，排序会变成：

1. `S3FT`
2. `ILR`
3. `SPPO`
4. `RISE`

## 2. ILR

论文：`Iterative Label Refinement Matters More than Preference Optimization under Weak Supervision`  
会议：`ICLR 2025 Spotlight`  
链接：<https://openreview.net/forum?id=q5EZ7gKcnW>

### 2.1 它到底在做什么

ILR 的出发点很强，也很现实：

传统后训练默认流程是：

`SFT -> 比较反馈 -> DPO/RLHF`

但这条路线有个隐藏前提：  
`人类 demonstration 和 preference feedback 基本可靠。`

ILR 直接挑战这个前提。它研究的是：

`如果监督本身不可靠，继续做 preference optimization 还有没有用？`

论文结论很鲜明：

- 在弱监督场景下，`SFT` 还有一定效果
- 但 `DPO` 往往很难稳定超过 SFT
- 这时更好的做法不是继续优化模型，而是用比较反馈去修训练标签

它的具体做法是：

1. 对每个训练样本，拿人类 demonstration 和模型新生成答案做比较
2. 用 preference feedback 判断是否该替换原标签
3. 得到更新后的训练集
4. 再重新做 SFT
5. 反复迭代

所以它不是“换一个新 loss”，而是：

`把后训练的重心从 model optimization 转到 data refinement。`

### 2.2 真正的创新点

ILR 的创新不在公式多漂亮，而在问题重构：

1. `挑战了 RLHF/DPO 的默认范式`
   它不是问“DPO loss 怎么改”，而是问“比较反馈是不是该拿来修数据，而不是修参数”。

2. `把偏好反馈从训练目标变成数据清洗信号`
   这是很重要的视角变化，尤其适合复杂任务和弱监督任务。

3. `特别适合复杂任务`
   数学、代码、安全对齐这些任务里，人类 demonstration 质量本来就不稳定，这正是 ILR 强的地方。

### 2.3 这个项目好在哪

- `故事非常完整`
  你能讲一个很清晰的问题链条：
  “监督不可靠 -> DPO 不稳 -> 修数据比修模型更有效”

- `不是纯调 loss`
  面试官通常更容易记住“这人在做数据闭环”，而不是“这人又换了个 preference loss”。

- `容易做出自己的分析`
  你可以加噪声实验、数据替换率统计、不同 judge 质量对结果的影响，这些都很有内容。

- `算力要求相对适中`
  不需要像 RLVR 那样长时间 rollout，也不需要像 self-play 那样多轮采样。

### 2.4 它的短板

- `看起来没有那么“炫”`
  它不是 RL，不是 verifier，不是 Nash equilibrium，表面上没有那么炸裂。

- `对实验设计要求高`
  你要把“弱监督”构造得合理，否则项目会变成一个很普通的数据清洗实验。

- `如果只跑 authors setting，工程亮点可能不够`
  所以最好要补充：
  噪声强度变化
  不同 judge
  label replacement 策略

### 2.5 复现可行性

可行性判断：`高`

原因：

- 训练管线接近普通 SFT/DPO
- 关键新增模块是 `candidate generation + preference-based relabeling`
- 任务可以缩到单一领域，比如 `GSM8K`、`MBPP`、`安全问答`

主要成本：

- 数据构造
- judge 或 comparison signal
- 多轮迭代训练管理

### 2.6 我对它的判断

项目质量：`很高`  
复现难度：`中`  
简历价值：`很高`

一句话评价：

`这是一个非常适合做成“有判断力的研究复现项目”的题，尤其适合你想体现自己对后训练范式有理解，而不只是会跑训练脚本。`

## 3. SPPO

论文：`Self-Play Preference Optimization for Language Model Alignment`  
会议：`ICLR 2025 Poster`  
链接：<https://openreview.net/forum?id=a3PmRgAB5T>

### 3.1 它到底在做什么

SPPO 解决的问题是：

`为什么偏好优化必须只看静态 pairwise 数据？能不能让模型自己和自己打，逐步逼近更强策略？`

它把对齐问题建模成一个 `constant-sum two-player game`，目标是找到 `Nash equilibrium policy`。

直观理解就是：

- 模型围绕同一批 prompt 采样多个答案
- 用偏好模型或打分器比较这些答案
- 再做自博弈式更新

所以它的本质不是普通 DPO，而是：

`self-play + preference optimization + game-theoretic alignment`

### 3.2 真正的创新点

1. `从静态偏好学习转成动态自博弈`
   这和普通 DPO 最大的差别在于，训练数据可以持续变成 on-policy 的。

2. `用博弈论解释偏好对齐`
   不是只做一个 heuristic loss，而是把目标写成逼近 Nash equilibrium。

3. `尽量减少外部监督依赖`
   论文强调不用额外更强教师做复杂标注，这一点很有吸引力。

### 3.3 这个项目好在哪

- `研究感最强`
  如果你想让项目显得像“真正做过一个 alignment 研究系统”，SPPO 很合适。

- `系统完整`
  它不是单一 loss 改动，而是一整个闭环：
  prompt -> sampling -> ranking -> update -> iterate

- `很容易画架构图`
  对简历和面试展示非常友好。

- `能衍生很多扩展`
  比如：
  prompt 难度调度
  PairRM 质量影响
  self-play 轮数
  response group size

### 3.4 它的短板

- `工程量明显更大`
  这不是一个轻松的周末项目。

- `实验细节容易卡住`
  例如采样策略、打分器质量、数据去重、on-policy 分布漂移，任何一项都可能影响结果。

- `如果算力一般，完整复现压力较大`
  因为它天然要做更多采样和更多轮训练。

### 3.5 复现可行性

可行性判断：`中`

原因：

- 方法清楚
- 代码思路相对完整
- 但要真正做得像样，需要把多个模块接好

最容易落地的复现方式：

1. 先选一个 7B 或更小底模
2. 只做单域任务或缩小 prompt 集
3. 用现成偏好模型打分
4. 跑一个 `mini self-play pipeline`

如果你想做成完整简历项目，最好补两个分析：

- self-play 的收益来自哪里
- 它和静态 DPO 的样本效率差异

### 3.6 我对它的判断

项目质量：`非常高`  
复现难度：`中高到高`  
简历价值：`非常高`

一句话评价：

`这是四个里最像“正式研究工程项目”的题，但你得愿意投入更多时间和算力。`

## 4. RISE

论文：`Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards`  
会议：`NeurIPS 2025 Poster`  
链接：<https://openreview.net/forum?id=gA3fFAEXNT>

### 4.1 它到底在做什么

RISE 属于 2025 非常新的主线：`RLVR`

它要解决的问题是：

`推理模型会做“表面反思”，但不一定真的会验证自己的答案。能不能把 self-verification 和 RL 一起训起来？`

它的核心思路是：

- 模型不仅要生成答案
- 还要生成对自己答案的验证或反思
- 再结合可验证奖励做在线 RL

所以 RISE 关注的不只是“答对”，而是：

`模型是否学会了更可靠的自验证行为。`

### 4.2 真正的创新点

1. `把 reasoning 和 self-verification 联合训练`
   不是把 verifier 当外部固定工具，而是把“会不会检查自己”也当作能力目标。

2. `站在 RLVR 的潮头上`
   这和传统 RLHF 差别很大，因为监督信号来自可验证奖励，而不是人类偏好。

3. `更贴近 2025 下半年的主流趋势`
   如果你想做“最近”的东西，RISE 比 ILR 和 SPPO 更新。

### 4.3 这个项目好在哪

- `很新`
  时间上很占优势，做出来会显得你跟得上 2025 的 reasoning post-training 趋势。

- `方向很热`
  verifier、RLVR、self-verification 都是目前非常主流的关键词。

- `适合讲能力提升机制`
  你可以讨论：
  模型到底是在学推理，还是在学检查
  verifier 能否减少 reward hacking

### 4.4 它的短板

- `复现门槛高`
  在线 RL、本地 rollout、可验证奖励、reasoning benchmark，这几个叠在一起就不轻。

- `很吃训练与评测基础设施`
  要做得规范，通常需要较成熟的 RL 框架和评测脚本。

- `如果只做小规模复现，故事可能会被削弱`
  因为这种论文的亮点常常依赖足够长的 RL 训练和足够难的 reasoning 任务。

### 4.5 复现可行性

可行性判断：`中低`

什么时候可行：

- 你有比较稳定的 RL 训练经验
- 你愿意把任务缩到数学或代码中的一个子集
- 你接受“复现思想 + 小规模验证”，而不是完整追论文主结果

更现实的做法是：

- 不追 full-scale RISE
- 只做 `self-verification head / format + RLVR` 的缩小版
- 在 `GSM8K / MATH subset / code correctness` 上验证

### 4.6 我对它的判断

项目质量：`高`  
复现难度：`高`  
简历价值：`高到非常高`

一句话评价：

`这是最贴近 2025 新热点的题，但如果资源一般，它更适合做“方向展示型项目”，不太适合作为第一个重型复现。`

## 5. S3FT

论文：`Selective Self-to-Supervised Fine-Tuning for Generalization in Large Language Models`  
会议：`Findings of NAACL 2025`  
链接：<https://aclanthology.org/2025.findings-naacl.349/>

### 5.1 它到底在做什么

S3FT 关注的是一个很常见但经常被忽略的问题：

`普通 SFT 往往会让模型在目标任务上变强，但在通用能力上变窄。`

它的核心做法很直白：

1. 先让模型自己对训练输入作答
2. 用 judge 选出其中“足够正确”的模型回答
3. 对这些样本，不再强行只学人工 gold answer，而是学模型自己的正确答案
4. 对剩余样本，再用 gold response 或其 paraphrase 去训练

本质上它是在利用：

`一个问题通常不只一个正确答案。`

### 5.2 真正的创新点

1. `用模型自己的正确答案降低过拟合`
   这比普通 SFT 更接近“保留底模分布”的思路。

2. `很实用`
   它没有引入特别重的训练框架，创新点很工程化、很可落地。

3. `把 generalization 作为核心目标`
   不是只追 target task 分数，而是看 fine-tuning 后通用基准掉不掉。

### 5.3 这个项目好在哪

- `最容易复现`
  它离标准 SFT 很近，改动小。

- `很适合做稳妥的简历项目`
  你能比较容易拿到成型结果。

- `适合做系统分析`
  例如：
  judge 精度影响
  self answer 占比
  不同 paraphrase 策略
  通用能力保留情况

### 5.4 它的短板

- `冲击力不如前 3 个`
  听起来没有 RL、self-play、verifier 那么吸睛。

- `会议级别略弱`
  是 `Findings of NAACL 2025`，不是 ICLR/NeurIPS 主会。

- `需要把项目包装好`
  如果你只是复现原文，很容易被看成“一个不错的数据配方改进”。

### 5.5 复现可行性

可行性判断：`很高`

原因：

- 和普通 SFT 高度兼容
- 不需要 RL
- 不需要复杂 preference pipeline
- 更适合小团队、个人项目

非常适合的项目形式：

- 选一个 reasoning 或 code 数据集
- 跑 `SFT vs S3FT`
- 同时看 in-domain 和 general benchmarks

### 5.6 我对它的判断

项目质量：`中高`  
复现难度：`低到中`  
简历价值：`中高`

一句话评价：

`这是四个里面最稳、最容易做出来的题，但它更像“高质量工程型项目”，没有 ILR 或 SPPO 那么强的研究辨识度。`

## 6. 四个项目横向比较

| 项目 | 核心主题 | 新颖度 | 工程复杂度 | 算力要求 | 讲故事能力 | 适合简历 | 我的总体判断 |
|---|---|---|---|---|---|---|---|
| ILR | 弱监督下的数据修正 | 高 | 中 | 中 | 很强 | 很强 | 最平衡 |
| SPPO | Self-play 偏好对齐 | 很高 | 高 | 高 | 很强 | 很强 | 最像研究项目 |
| RISE | RLVR + self-verification | 很高 | 很高 | 很高 | 强 | 很强 | 最新，但重 |
| S3FT | 泛化导向 SFT | 中 | 低 | 低到中 | 中 | 中高 | 最稳妥 |

## 7. 如果你现在就要做决定

### 选 ILR，如果你想要：

- 有观点
- 有实验空间
- 不想把项目做成纯调 loss
- 希望复现难度和价值比较平衡

### 选 SPPO，如果你想要：

- 一个完整的大项目
- 强研究味
- 愿意投入更多时间和算力

### 选 RISE，如果你想要：

- 紧贴 2025 最新热点
- 做 reasoning / verifier / RLVR
- 接受高风险高成本

### 选 S3FT，如果你想要：

- 尽快做出稳定结果
- 项目风险低
- 从 SFT 入手做一个干净扎实的项目

## 8. 最后的建议

如果只能选一个，我建议优先顺序是：

1. `ILR`
2. `SPPO`
3. `S3FT`
4. `RISE`

原因不是 RISE 不好，而是：

`RISE 更前沿，但对个人复现项目来说成本偏高；ILR 和 SPPO 更容易做出“我真的掌握了这个方向”的成果。`

如果你准备做两个项目，那么最好的组合是：

- `ILR + S3FT`
  一个偏数据中心后训练，一个偏稳健 SFT，很互补

或者：

- `SPPO + RISE`
  一个偏 preference optimization，一个偏 RLVR/verifier，研究味最足

