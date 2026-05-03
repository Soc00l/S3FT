# S3FT 可执行任务清单

更新时间：2026-05-03  
用途：把 [s3ft_project_plan.md](</C:/Users/彭富/Documents/New project/s3ft_project_plan.md>) 压缩成能直接开工的执行清单。  
默认目标：先做一个 `数学推理` 版本，优先追求 `闭环完整 + 结论清楚`，不一开始追大模型和大全量。

## 1. 先定项目边界

第一版不要贪大，先锁死下面的范围：

- 任务：`GSM8K -> GSM8K / SVAMP / MATH subset`
- 模型：`Qwen2.5-1.5B` 或 `Qwen2.5-3B`
- 方法对比：
  - `Vanilla SFT`
  - `S3FT`
  - `Self-answer only`
- judge：`exact answer matching + 数值标准化`

第一版先不要做：

- 开放式对话
- LLM-as-a-judge
- DPO 二阶段
- 多模型横向大扫荡

目标很简单：

`先把 S3FT 的最小研究闭环跑通，再决定要不要扩。`

## 2. 目录和代码结构

建议你在当前工程里先补这些文件或目录：

1. `data/`
   放原始数据、处理中间产物、重写后的训练集

2. `scripts/`
   放数据处理、生成 self-answer、judge、评测脚本

3. `configs/`
   放训练配置

4. `results/`
   放日志、指标、样例分析

5. `reports/`
   放实验记录和图表

建议的最小文件集合：

- `scripts/prepare_gsm8k.py`
- `scripts/generate_self_answers.py`
- `scripts/build_s3ft_dataset.py`
- `scripts/eval_math.py`
- `configs/sft_baseline.yaml`
- `configs/s3ft.yaml`
- `configs/self_answer_only.yaml`

## 3. 第一周：先跑通最小闭环

这周的目标不是出漂亮结果，而是把全流程串起来。

### Task 1：数据准备

要做的事：

1. 下载或整理 `GSM8K` 训练/测试集
2. 统一成训练脚本可读格式，例如：
   - `prompt`
   - `response`
   - `answer`
3. 把最终答案抽出来，确保 judge 可以直接比

完成标准：

- 有一个干净的 `train.jsonl`
- 有一个干净的 `test.jsonl`
- 你能随机抽样 20 条看懂格式

产出物：

- `data/gsm8k/train.jsonl`
- `data/gsm8k/test.jsonl`

### Task 2：Vanilla SFT baseline

要做的事：

1. 选一个最熟的训练框架
   - `transformers + trl`
   - `LLaMA-Factory`
   - 或你自己的脚本
2. 跑一个小规模 baseline
3. 在 `GSM8K test` 上验证 accuracy

完成标准：

- baseline 能稳定训练完
- 有可复现实验配置
- 能拿到第一版测试指标

产出物：

- `configs/sft_baseline.yaml`
- `results/baseline_metrics.json`

### Task 3：self-answer 生成脚本

要做的事：

1. 用 baseline 模型对训练集 prompt 生成答案
2. 保存：
   - 原始 prompt
   - gold response
   - model response
   - extracted final answer
3. 先只跑一个子集，比如 `2k-5k` 条

完成标准：

- 生成脚本能稳定断点续跑
- 输出结构固定

产出物：

- `scripts/generate_self_answers.py`
- `data/gsm8k/self_answers_subset.jsonl`

### Task 4：最简 judge

要做的事：

1. 实现数值答案抽取
2. 做标准化比较：
   - 去空格
   - 去货币符号
   - 处理整数/小数格式
3. 标记：
   - `is_correct`
   - `judge_reason`

完成标准：

- 随机抽 100 条人工 spot check
- judge precision 尽量高

产出物：

- `scripts/eval_math.py`
- `data/gsm8k/self_answers_scored_subset.jsonl`

### Task 5：构建第一版 S3FT 数据

要做的事：

1. 如果 self-answer 正确，就替换 gold response
2. 如果 self-answer 错误，就保留 gold response
3. 导出新训练集

完成标准：

- 能统计替换比例
- 能抽样查看替换样本是否合理

产出物：

- `scripts/build_s3ft_dataset.py`
- `data/gsm8k/train_s3ft_subset.jsonl`

## 4. 第二周：跑出主结果

这周开始正式做对比。

### Task 6：训练三条主线

你至少要跑：

1. `Vanilla SFT`
2. `S3FT`
3. `Self-answer only`

如果时间够，再加：

4. `Gold + paraphrase`

完成标准：

- 四条线都能产出统一格式的 checkpoint 和 eval 结果

产出物：

- `results/main_table_v1.csv`

### Task 7：统一评测

至少评：

- `GSM8K`
- `SVAMP`
- `MATH subset`

要保证：

- prompt 模板一致
- decoding 设置一致
- 提取答案逻辑一致

完成标准：

- 每个模型都有一张统一表格

### Task 8：先写第一版结论

这一步很重要，不要等全部实验做完才写。

你要先回答：

1. `S3FT` 是否优于 `Vanilla SFT`
2. `Self-answer only` 是否明显不稳
3. 替换比例大概是多少
4. out-of-domain 上有没有初步提升

产出物：

- `reports/week2_findings.md`

## 5. 第三周：把项目做厚

这一周重点不是再多跑一堆主实验，而是做最值钱的分析。

### Task 9：Judge 质量分析

做法：

1. 人工抽样 100-200 条
2. 检查 judge 的：
   - false positive
   - false negative
3. 统计误差模式

如果有余力，可做一个弱 judge 对照：

- 严格 exact match
- 宽松数值 match

你最终要回答：

`S3FT 对 judge 噪声有多敏感。`

产出物：

- `reports/judge_error_analysis.md`

### Task 10：替换样本分析

要统计：

- 替换比例
- 替换样本的题目难度
- self-answer 与 gold 的长度差异
- self-answer 与 gold 的形式差异

你最终要回答：

`模型在哪类题上更容易提供“可学的正确替代答案”。`

产出物：

- `results/rewrite_stats.csv`

### Task 11：best-of-n / self-consistency 小实验

这是很加分的一步。

做法：

1. 对 `Vanilla SFT` 和 `S3FT` 分别采样 `n=5` 或 `n=10`
2. 比较：
   - best-of-n oracle upper bound
   - self-consistency majority vote

你最终要回答：

`S3FT 是否保留了更有利用价值的答案分布。`

产出物：

- `results/test_time_scaling.csv`

## 6. 第四周：可选增强

如果前三周结果已经不错，这一周可以做一个加分项。

### 选项 A：Gold + paraphrase baseline

目的：

- 排除“只是答案表面变多样”这个解释

### 选项 B：S3FT -> DPO 小规模二阶段

目的：

- 验证 S3FT 是否更适合做后续对齐冷启动

### 选项 C：再加一个任务

比如：

- 再加一个数学集
- 或转到 `MBPP` 做小规模代码实验

这周不建议三件都做，选一个最能抬高项目价值的就够了。

优先级建议：

1. `best-of-n` 已经做过的话，优先 `Gold + paraphrase`
2. 如果你想把项目和后训练更强绑定，优先 `S3FT -> DPO`
3. 如果你想做更泛化的论断，再加第二任务

## 7. 每周都要维护的东西

这些最容易被忽略，但决定你项目最后是不是像样。

### 实验日志

每次训练都记录：

- 模型名
- 数据版本
- 配置文件
- 学习率
- epoch
- batch size
- seed
- eval 指标

建议维护：

- `reports/experiment_log.md`

### 样例池

每周都保存一些典型样例：

- S3FT 明显比 SFT 好的
- S3FT 失败的
- judge 判错的
- self-answer 替换 gold 很有意思的

建议维护：

- `reports/case_studies.md`

这些素材后面写博客、写简历、面试举例都非常有用。

## 8. 最低可交付版本

如果时间不够，最少也要完成下面这些：

1. `Vanilla SFT / S3FT / Self-answer only`
2. `GSM8K + 1 个 out-of-domain 集`
3. `替换比例分析`
4. `judge 误差分析`
5. `一页总结报告`

做到这一步，这个项目就已经比普通“复现一个 SFT trick”厚很多了。

## 9. 最佳版本

如果一切顺利，最佳版本是：

1. 三到四条 baseline
2. 三个评测集
3. judge 质量分析
4. 替换样本统计
5. best-of-n 或 self-consistency
6. 一个二阶段后训练扩展

做到这里，项目已经很像一篇小型研究复现了。

## 10. 开工顺序

如果你明天就开始，顺序就按这个来：

1. 整理 `GSM8K` 数据格式
2. 跑通 `Vanilla SFT`
3. 写 self-answer 生成脚本
4. 写数值 judge
5. 构建 `S3FT` 数据集
6. 跑 `S3FT` 和 `Self-answer only`
7. 做 `SVAMP / MATH subset` 评测
8. 做 judge 和替换样本分析
9. 再决定要不要上 `best-of-n` 或 `DPO`

## 11. 最后提醒

这个项目最容易犯的错有两个：

1. 一上来就铺太大，结果 baseline 都没跑稳
2. 只盯最终 accuracy，不做替换样本和 judge 分析

你真正该追求的是：

`用一个轻方法，做出一组很有解释力的实验。`

