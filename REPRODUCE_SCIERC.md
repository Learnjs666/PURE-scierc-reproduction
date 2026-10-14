# PURE (SciERC) 论文复现与端到端评测实录

> **论文信息**：*A Frustratingly Easy Approach for Entity and Relation Extraction* (NAACL 2021)  
> **论文作者**：Zexuan Zhong (钟泽轩), Danqi Chen (陈丹琦) (Princeton University)  
> **核心机制**：流水线解耦架构（Decoupled Pipeline）、带类型实体标记（Typed Entity Markers）、跨度级别表示（Span Representations）。  
> **复现基准**：SciERC 科技文献抽取基准测试集 (551 句测试集, 974 个黄金关系三元组)。

---

## 目录
1. [现代深度学习工程改造点说明 (Modernization Changelog)](#1-现代深度学习工程改造点说明-modernization-changelog)
2. [环境准备与依赖安装](#2-环境准备与依赖安装)
3. [权重下载与数据准备](#3-权重下载与数据准备)
4. [三步复现核心命令与操作指南](#4-三步复现核心命令与操作指南)
   - [第一步：实体识别预测 (Stage 1: Entity Span Prediction)](#第一步实体识别预测-stage-1-entity-span-prediction)
   - [第二步：关系抽取预测 (Stage 2: Relation Extraction)](#第二步关系抽取预测-stage-2-relation-extraction)
   - [第三步：端到端评测与论文对齐 (Stage 3: End-to-End Evaluation)](#第三步端到端评测与论文对齐-stage-3-end-to-end-evaluation)
5. [实测结果与论文 Table 1 精准对照](#5-实测结果与论文-table-1-精准对照)
6. [REL 与 REL Strict 评测机理与级联误差剖析](#6-rel-与-rel-strict-评测机理与级联误差剖析)
7. [单句交互式演示与注意力掩码可视化 (Demo Pipeline)](#7-单句交互式演示与注意力掩码可视化-demo-pipeline)
8. [一键执行脚本使用说明](#8-一键执行脚本使用说明)

---

## 1. 现代深度学习工程改造点说明 (Modernization Changelog)

原论文开源代码基于 2020 年的 PyTorch 1.x、Transformers 2.x 以及强依赖已废弃的 `allennlp` 库。为使本项目能够在现代 Python 3.9+、PyTorch 2.x 和 HuggingFace Transformers 4.x 环境下平稳运行，并彻底摆脱陈旧依赖冲突，本项目进行了**彻底的白盒化工程重构**：

1. **剔除 `allennlp` 依赖，纯原生 PyTorch 重写**：
   - 在 `entity/models.py` 与 `relation/models.py` 中重写了 `FeedForward` 和 `batched_index_select` 算子，结构与官方预训练权重的 `state_dict` 保持 100% 精确对齐。
2. **修复 HuggingFace 优化器迁移冲突**：
   - 将已弃用的 `from transformers import AdamW` 迁移为现代官方标准的 `from torch.optim import AdamW`。
3. **修复 BertPreTrainedModel 权重初始化机制**：
   - 增加 `hasattr(self, 'post_init')` 动态探测，兼容现代 Transformers 的模块化初始化逻辑，避免权重覆盖与绑定张量报警。
4. **张量解包兼容与 CPU/GPU 自适应回退**：
   - 将旧版元组解包统一对齐为 `outputs[0]`；在 `run_entity.py` 中增加自动 CPU 回退机制，无显卡环境下亦可顺畅推理。
5. **修复种子设置 Bug**：
   - 修正 `run_relation.py` 中 `np.random.seed(args.seed)` 的变量名笔误为 `np.random.seed(seed)`。

---

## 2. 环境准备与依赖安装

建议使用虚拟环境（如已在当前目录创建 `venv`）：

```bash
# 激活虚拟环境 (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 核心依赖项说明 (已验证版本兼容性)
# torch >= 2.0.0
# transformers >= 4.20.0
# numpy
# tqdm
```

---

## 3. 权重下载与数据准备

本项目提供了自动下载并解压官方预训练模型的脚本：

```bash
# 自动拉取官方 ent-scib-ctx0 与 rel-scib-ctx0 权重包并解压至 scierc_models/
python download_models.py
```

- **数据位置**：`scierc_data/processed_data/json/` 包含预处理完成的 SciERC 语料：
  - `train.json` (训练集, 1,861 句)
  - `dev.json` (验证集, 275 句)
  - `test.json` (测试集, 551 句)
- **模型位置**：
  - `scierc_models/ent-scib-ctx0/` (基于 SciBERT 的实体跨度预测模型)
  - `scierc_models/rel-scib-ctx0/` (基于 SciBERT 的 Typed Marker 关系抽取模型)

---

## 4. 三步复现核心命令与操作指南

### 第一步：实体识别预测 (Stage 1: Entity Span Prediction)

该阶段基于单句中长度 $\le 8$ 的候选跨度进行穷举，输入 SciBERT 编码后通过 MLP 分类器预测实体类别。

#### 执行命令 (PowerShell)
```powershell
python run_entity.py `
  --do_eval `
  --eval_test `
  --context_window 0 `
  --task scierc `
  --data_dir scierc_data/processed_data/json/ `
  --model allenai/scibert_scivocab_uncased `
  --output_dir scierc_models/ent-scib-ctx0/
```

#### 参数解析：
- `--do_eval --eval_test`：仅运行测试集评估（Inference 模式）。
- `--context_window 0`：不拼接待测试句子前后的跨句上下文（与论文主实验单句设定对齐）。
- `--task scierc`：指定任务标签集（Method, OtherScientificTerm, Task, Generic, Material, Metric）。
- `--output_dir`：预测结果输出目录。

#### 产物说明：
- 输出文件：`scierc_models/ent-scib-ctx0/ent_pred_test.json`
- 该文件包含了测试集 551 句文本中识别出的 1,680 个实体跨度预测（含边界 `[start, end]` 及预测类别）。

---

### 第二步：关系抽取预测 (Stage 2: Relation Extraction)

该阶段读取第一阶段预测出的实体跨度，动态在句子中注入 Typed Entity Markers（如 `<S:Method> ... </S:Method>` 和 `<O:Material> ... </O:Material>`），枚举所有候选实体对，预测其关系类型。

#### 执行命令 (PowerShell)
```powershell
python run_relation.py `
  --task scierc `
  --do_eval `
  --eval_test `
  --model allenai/scibert_scivocab_uncased `
  --do_lower_case `
  --context_window 0 `
  --max_seq_length 128 `
  --entity_output_dir scierc_models/ent-scib-ctx0/ `
  --output_dir scierc_models/rel-scib-ctx0/
```

#### 参数解析：
- `--entity_output_dir`：指定第一阶段实体模型生成的预测目录（程序会自动读取 `ent_pred_test.json` 作为输入）。
- `--max_seq_length 128`：输入序列的最大截断长度。
- `--output_dir`：关系预测结果保存目录。

#### 产物说明：
- 输出文件：`scierc_models/rel-scib-ctx0/predictions.json`
- 历经 5,092 对候选实体对前向计算，共预测出 954 个有效关系，其中 469 个完全命中。

---

### 第二步（B）：近似批处理关系抽取预测 (Stage 2-Approx: Fast Approximation Relation Model)

针对全量模型需要将 5,092 对候选实体拆分为 5,092 个独立句子做前向计算的 $\mathcal{O}(k^2)$ 开销瓶颈，近似模型将同一句内的多对 Typed Markers 追加至句末，通过**位置绑定（Position Tieing）**与**二维非对称定向注意力掩码（Directed Attention Mask）**，将 5,092 对候选关系打包压缩为 **470 个句子级样本**（平均每句打包 10.83 对），在单次前向中并行打分。

#### 执行命令 (PowerShell)
```powershell
python run_relation_approx.py `
  --task scierc `
  --do_eval `
  --eval_test `
  --model allenai/scibert_scivocab_uncased `
  --do_lower_case `
  --context_window 0 `
  --max_seq_length 250 `
  --entity_output_dir scierc_models/ent-scib-ctx0/ `
  --output_dir scierc_models/rel_approx-scib-ctx0/ `
  --batch_computation
```

#### 产物说明：
- 输出文件：`scierc_models/rel_approx-scib-ctx0/predictions.json`
- 前向样本数从 `5,092` 骤降至 **`470`**（压缩比 **10.83 倍**），CPU 推理耗时仅 **`62.327 s`**，预测出 884 个有效关系（442 个命中，精确率达 50.00%）。

---

### 第三步：端到端评测与论文对齐 (Stage 3: End-to-End Evaluation)

该阶段调用端到端评测脚本，同时计算实体识别评测指标与关系抽取评测指标（包括 Boundary Matching 与 Strict Matching）。

#### 执行命令 (PowerShell)
```powershell
# 评测标准全量关系模型
python run_eval.py --prediction_file scierc_models/rel-scib-ctx0/predictions.json

# 评测近似批处理关系模型
python run_eval.py --prediction_file scierc_models/rel_approx-scib-ctx0/predictions.json
```

---

## 5. 实测结果与论文 Table 1 / Table 5 精准对照

评测脚本执行后，核心评测输出如下：

```text
=== 1. Full Relation Model (rel-scib-ctx0, 5092 examples) ===
  NER          - P: 0.667857, R: 0.665875, F1: 0.666865
  REL          - P: 0.491614, R: 0.481520, F1: 0.486515
  REL (strict) - P: 0.360587, R: 0.353183, F1: 0.356846

=== 2. Approximation Relation Model (rel_approx-scib-ctx0, 470 examples, 62.327 s) ===
  NER          - P: 0.667857, R: 0.665875, F1: 0.666865
  REL          - P: 0.500000, R: 0.453799, F1: 0.475780
  REL (strict) - P: 0.376697, R: 0.341889, F1: 0.358450
```

### 论文 Table 1 & Table 5 (End-to-end & Approximation Results on SciERC) 严格对齐表：

| 评估指标 (Metric) | 评测匹配准则 (Criteria) | Full Model 实测值 (`5092` 次前向) | Approx Model 实测值 (`470` 次前向, `62.3s`) | 论文汇报基线值 (无跨句上下文) | 对齐结论与状态 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NER F1** | 跨度起止边界 + 实体类别完全正确 | **66.69%** | **66.69%** | 66.6% | ✅ **完美对齐** (偏差 +0.09%) |
| **REL F1** | 跨度起止边界 + 关系类别匹配 (忽略实体类型) | **48.65%** | **47.58%** | 48.2% | ✅ **完美对齐** (Approx 仅差 ~1%) |
| **REL Strict F1** | 跨度起止边界 + **实体类别** + 关系类别全部匹配 | **35.68%** | **35.85%** | 35.6% | ✅ **完美对齐** (Approx 甚至反超 +0.17%!) |

---

## 6. REL 与 REL Strict 评测机理与级联误差剖析

在端到端信息抽取任务中，REL 与 REL Strict 存在高达 **~13%** 的指标差距（48.65% vs 35.68%），这深刻揭示了流水线（Pipeline）模型的经典特质与级联误差机制：

### 1. 两者的评测判决公式区别
- **REL (Boundary Matching)**：
  一个三元组 $(e_1, r, e_2)$ 判定为正确，充要条件是：
  $$\text{Span}(e_1) = \text{Span}(e_1^*) \quad \land \quad \text{Span}(e_2) = \text{Span}(e_2^*) \quad \land \quad r = r^*$$
  **不论 $e_1$ 和 $e_2$ 的实体类型预测是否正确**，只要实体边界和两者的语义关系预测正确，即判定为命中。
- **REL Strict (Strict Matching)**：
  必须满足三位一体严格一致：
  $$\text{Span}(e_1) = \text{Span}(e_1^*) \land \text{Type}(e_1) = \text{Type}(e_1^*) \land \text{Span}(e_2) = \text{Span}(e_2^*) \land \text{Type}(e_2) = \text{Type}(e_2^*) \land r = r^*$$

### 2. 级联噪声鲁棒性与误差放大
第一阶段 NER 的准确率为 66.69%，意味着有约 33.3% 的实体类型可能预测有误。
- 当一个实体的真实类别为 `Method`，却被第一阶段误识别为 `OtherScientificTerm` 时：
  - 关系模型在第二阶段接收到的输入是带噪声的标记 `<S:OtherScientificTerm>`；
  - 尽管输入标记的类型存在噪声，但模型凭借句子上下文词汇（如 "is a parser for"），**依然极其稳健地将两者关系判定为正确的 `USED-FOR`**！
  - 在 **REL** 准则下，由于实体边界和关系均正确，该预测被计为正确命中（TP）；
  - 但在 **REL Strict** 准则下，由于实体类型标签错位，该预测被直接判定为错误。这就是 13% 差距的根源。

---

## 7. 单句交互式演示与注意力掩码可视化 (Demo Pipeline)

为了在没有任何黑盒封装的情况下直观观测数据流和模型内部机制，本项目提供了极简单句演示脚本 `demo_pipeline.py`：

```bash
python demo_pipeline.py
```

### 脚本核心功能：
1. **纯原生 CPU 极简推理**：不依赖任何复杂框架，仅用纯 PyTorch 实例化模型；
2. **完整复现论文 Figure 1 示例句**：
   - 句子：`["MORPA", "is", "a", "fully", "implemented", "parser", "for", "text-to-speech"]`
   - 展示跨度枚举过程与置信度打分；
   - 展示带有 Typed Entity Markers 的重构输入序列；
   - 输出最终关系三元组抽取结果。
3. **二维定向注意力掩码矩阵可视化 (Attention Mask Visualizer)**：
   - 直观打印论文 Figure 1(c) 核心机制矩阵：解释模型如何通过非对称注意力掩码将 $\mathcal{O}(k^2)$ 次前向压缩为 1 次前向批处理计算，而文本 Token 绝不受到任何 Marker 的反向串扰。

---

## 8. 一键执行脚本使用说明

在 Windows PowerShell 环境下，可直接运行根目录提供的自动化脚本完成全流程评测：

```powershell
.\run_reproduction.ps1
```

该脚本将依次自动化执行：
1. 检查虚拟环境与模型权重；
2. 执行第一阶段实体预测；
3. 执行第二阶段关系预测；
4. 运行端到端结果评测并输出对齐汇总。
