# MatSciBERT + PURE (SuperMat) 落地与端到端实操指南

> **任务定位**：在现有 PURE 现代化复现架构之上，以材料科学专用预训练语言模型 **MatSciBERT** 为基座，在超导材料与物性权威基准 **SuperMat** 上完成端到端实体与关系抽取流水线的构建、微调与严格评测。  
> **核心原则**：模块化正交隔离、拒绝工程臃肿、复用深度学习算子底座、全流程白盒透明。

---

## 目录
1. [模块化目录规划与隔离设计](#1-模块化目录规划与隔离设计)
2. [SuperMat 领域实体与关系 Schema 规范](#2-supermat-领域实体与关系-schema-规范)
3. [数据预处理四大避坑铁律](#3-数据预处理四大避坑铁律)
4. [Schema 注册与代码接入 (`shared/const.py`)](#4-schema-注册与代码接入-sharedconstpy)
5. [分步执行指令与超参数配置](#5-分步执行指令与超参数配置)
   - [第一步：数据集准备与预训练权重缓存](#第一步数据集准备与预训练权重缓存)
   - [第二步：执行数据转换与标准化清洗](#第二步执行数据转换与标准化清洗)
   - [第三步：微调 Stage 1 实体抽取模型](#第三步微调-stage-1-实体抽取模型)
   - [第四步：微调 Stage 2 关系抽取模型 (Typed Markers)](#第四步微调-stage-2-关系抽取模型-typed-markers)
   - [第五步：端到端级联评估 (Strict F1 对齐)](#第五步端到端级联评估-strict-f1-对齐)
   - [第六步：SciBERT vs MatSciBERT 消融对照实验](#第六步scibert-vs-matscibert-消融对照实验)
6. [一键运行脚本 (`run_supermat.ps1`) 模板](#6-一键运行脚本-run_supermatps1-模板)

---

## 1. 模块化目录规划与隔离设计

为避免多任务实验导致项目杂乱，采用“**数据与模型正交隔离、核心网络与虚拟环境完全复用**”的目录架构：

```text
experiments/pure/
├── entity/                         # PURE 实体核心网络 (通用复用)
├── relation/                       # PURE 关系抽取核心网络 (通用复用)
├── shared/                         # 增量注册 task_ner_labels / task_rel_labels
│
├── scierc_data/                    # [基准域 A] 通用科技文献数据 (SciERC)
├── scierc_models/                  # [基准域 A] SciBERT 产出权重与日志
│
├── materials_data/                 # [基准域 B] 材料科学数据根目录
│   ├── raw/                        # 原始下载的 SuperMat 仓库与标注
│   └── json/                       # 转换后的 PURE 标准格式 (train/dev/test.json)
├── materials_models/               # [基准域 B] MatSciBERT 产出权重与评估报告
│   ├── ent-matscib-ctx300/         # 实体识别模型 Checkpoint
│   └── rel-matscib-ctx100/         # 关系抽取模型 Checkpoint
│
├── scripts/                        # 专用脚本独立归档 (避免根目录杂乱)
│   ├── convert_supermat_to_pure.py # SuperMat 格式解析与转换核心逻辑
│   └── normalize_text.py           # 化学式与特殊 Unicode 符号正则化
│
├── run_entity.py                   # 通用实体微调/推理入口
├── run_relation.py                 # 通用关系微调/推理入口
├── run_eval.py                     # 通用严格指标评估脚本
├── run_supermat.ps1                # SuperMat 流水线一键自动化运行脚本
├── run_reproduction.ps1            # SciERC 复现一键脚本
├── REPRODUCE_SCIERC.md             # SciERC 复现实录
└── SUPERMAT_MATSCIBERT_GUIDE.md    # 本指南
```

---

## 2. SuperMat 领域实体与关系 Schema 规范

SuperMat (Science and Technology of Advanced Materials: Methods, 2021) 聚焦于超导材料的组分、分类及其物理性能关联：

### 2.1 实体标签 (NER Schema - 6 类)
1. `material`：化学式、合金组分、单晶或掺杂标记（例如：`Ba0.95La0.05FeO3-δ`, `YBCO-123`, `x = 0.15`）。
2. `class`：超导体所属类别或家族（例如：`cuprates`, `iron-based`, `pnictide`）。
3. `tc`：临界温度文字引导词/符号（例如：`critical temperature`, `Tc`, `superconducting transition`）。
4. `tcValue`：实测临界温度数值与单位（例如：`39 K`, `∼92K`, `from 3 to 4 K`）。
5. `pressure`：实验施加的高压环境参数（例如：`ambient pressure`, `∼2 GPa`, `up to 9.4 GPa`）。
6. `me_method`：临界温度测定实验手段（例如：`resistivity`, `susceptibility`, `tunneling spectroscopy`）。

### 2.2 关系标签 (RE Schema - 3 类强定向二元三元组)
1. `material-tc`：连接超导临界温度与其对应材料（$tcValue \to material$）。
2. `tc-pressure`：连接临界温度与其测定时的压力条件（$pressure \to tcValue$）。
3. `me_method-tc`：连接临界温度与其测试表征手段（$tcValue \to me_method$）。

---

## 3. 数据预处理四大避坑铁律

PURE 框架解析器底层（`shared/data_structures.py`）对输入张量有严格的拓扑要求，编写转换脚本必须遵守以下规范：

1. **铁律一：全局文档级累加索引 (Document-level Cumulative Token Offset)**
   - PURE 中的 `[start, end]` 是在整篇 Document 所有句子展开后的**绝对索引**，严禁使用单句内的相对局部索引。
   - PURE 解析器读取到数据后，通过计算每句话的起始偏移 `sentence_starts = np.cumsum(sentence_lengths)`，利用 `start - sentence_start` 还原局部切片。传相对索引将导致数组切片负越界或指向错误词元。
2. **铁律二：闭区间边界原则 (Inclusive Boundary)**
   - PURE 的实体与关系跨度全部为闭区间 `[start, end]`（即 Python 切片 `tokens[start : end + 1]`）。
   - 单 token 实体在下标 5 时，必须标为 `[5, 5, "TYPE"]`，绝不能是左闭右开的 `[5, 6]`。
3. **铁律三：严格限定在句内关系 (Intra-sentence Relations)**
   - PURE 的 Stage 2 关系分类机制是将候选实体对注入到同一个句子上下文内构建 Typed Markers。跨句关系在 PURE 默认单句模式下无法构建 Attention Mask。转换时必须对跨句三元组进行**前置过滤**或使用上下文滑动窗口进行段落合并。
4. **铁律四：材料专业符号正则化 (Unicode Normalization)**
   - 材料化学式包含大量的 Unicode 下标（`₂`, `₃`, `₄`）、希腊字母（$\alpha, \beta, \delta$）、温度符号（`℃`, `°`）与特殊减号（`−`）。
   - 在喂入 Tokenizer 之前必须调用 `normalize_text.py` 映射回标准 ASCII。若直接分词，这些符号会被 SciBERT/MatSciBERT 的 WordPiece 分词器全部转为 `[UNK]`，导致化学式被粉碎性截断。

---

## 4. Schema 注册与代码接入 (`shared/const.py`)

在 `shared/const.py` 中注册 `supermat` 任务，使参数 `--task supermat` 能够正确加载 Label Map：

```python
# shared/const.py 增量修改对照

task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    # 【新增】SuperMat 实体体系
    'supermat': ['material', 'class', 'tc', 'tcValue', 'pressure', 'me_method'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    # 【新增】SuperMat 关系体系
    'supermat': ['material-tc', 'tc-pressure', 'me_method-tc'],
}
```

---

## 5. 分步执行指令与超参数配置

### 第一步：数据集准备与预训练权重缓存

1. 拉取 SuperMat 官方开源数据至隔离目录：
```powershell
mkdir -p materials_data/raw
git clone https://github.com/lfoppiano/SuperMat.git materials_data/raw/SuperMat
```

2. 预先拉取 MatSciBERT 权重至 Hugging Face 缓存（确保网络畅通）：
```powershell
.\venv\Scripts\Activate.ps1
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('m3rg-iitd/matscibert'); AutoModel.from_pretrained('m3rg-iitd/matscibert'); print('MatSciBERT Downloaded Successfully!')"
```

### 第二步：执行数据转换与标准化清洗
运行预处理脚本将 SuperMat 的 Brat / XML 语料清洗并转换为 PURE 标准 JSONL：
```powershell
python scripts/convert_supermat_to_pure.py `
  --input_dir materials_data/raw/SuperMat/ `
  --output_dir materials_data/json/ `
  --train_ratio 0.8 `
  --dev_ratio 0.1 `
  --test_ratio 0.1
```
> **检查输出**：确认 `materials_data/json/` 目录下生成 `train.json`、`dev.json`、`test.json`，且抽样检查单条数据具有 `doc_key`, `sentences`, `ner`, `relations` 四个字段。

### 第三步：微调 Stage 1 实体抽取模型
以 `m3rg-iitd/matscibert` 为骨干网络，微调材料跨度枚举分类器：
```powershell
python run_entity.py `
  --do_train `
  --do_eval `
  --eval_test `
  --task supermat `
  --data_dir materials_data/json/ `
  --model m3rg-iitd/matscibert `
  --output_dir materials_models/ent-matscib-ctx300/ `
  --learning_rate 1e-5 `
  --task_learning_rate 5e-4 `
  --train_batch_size 16 `
  --context_window 300 `
  --max_span_length 10
```
- **核心参数解析**：
  - `--model m3rg-iitd/matscibert`：指定材料专用 BERT 底座。
  - `--context_window 300`：拼接前后 300 词元的上下文信息，提升长句材料化学式边界判断力。
  - `--max_span_length 10`：材料化学式与测量方法常包含复合长词，将最大跨度从 8 调优放宽至 10。
- **阶段产物**：在 `materials_models/ent-matscib-ctx300/` 中产出 `ent_pred_test.json`（已附带 `predicted_ner` 预测跨度）。

### 第四步：微调 Stage 2 关系抽取模型 (Typed Markers)
利用实体模型产出的实体预测结果，通过注入类型标记微调关系分类模型：
```powershell
python run_relation.py `
  --do_train `
  --do_eval `
  --eval_test `
  --task supermat `
  --data_dir materials_data/json/ `
  --model m3rg-iitd/matscibert `
  --do_lower_case `
  --entity_output_dir materials_models/ent-matscib-ctx300/ `
  --output_dir materials_models/rel-matscib-ctx100/ `
  --learning_rate 2e-5 `
  --train_batch_size 32 `
  --max_seq_length 256 `
  --context_window 100
```
- **核心机制**：
  - 自动向词表映射 Typed Markers（例如：`[S:material]`, `[O:tcValue]` 映射至 `[unused1]~[unused20]`）。
  - `--entity_output_dir`：读取 Stage 1 的实体识别产物，实现解耦流水线串联。

### 第五步：端到端级联评估 (Strict F1 对齐)
评估真实端到端流水线性能，剖析级联误差：
```powershell
python run_eval.py --prediction_file materials_models/rel-matscib-ctx100/predictions.json
```
- **关注两组指标输出**：
  1. `REL F1`（基于黄金实体边界）：考核关系抽取头本身的分类判别能力。
  2. `REL Strict F1`（端到端级联评测）：实体主体、客体边界、实体类型、关系类型**四项必须全部命中黄金标准**，代表实际工业/学术落地的真实可用率。

### 第六步：SciBERT vs MatSciBERT 消融对照实验
为验证 MatSciBERT 在材料领域的真实增益，保持所有种子与超参数完全一致，仅将底座换为 `allenai/scibert_scivocab_uncased`：
```powershell
# 训练对照组 Entity 模型
python run_entity.py `
  --do_train --do_eval --eval_test `
  --task supermat `
  --data_dir materials_data/json/ `
  --model allenai/scibert_scivocab_uncased `
  --output_dir materials_models/ent-scib-ctx300/ `
  --learning_rate 1e-5 --task_learning_rate 5e-4 --train_batch_size 16 --context_window 300 --max_span_length 10

# 训练对照组 Relation 模型
python run_relation.py `
  --do_train --do_eval --eval_test `
  --task supermat `
  --data_dir materials_data/json/ `
  --model allenai/scibert_scivocab_uncased `
  --do_lower_case `
  --entity_output_dir materials_models/ent-scib-ctx300/ `
  --output_dir materials_models/rel-scib-ctx100/ `
  --learning_rate 2e-5 --train_batch_size 32 --max_seq_length 256 --context_window 100
```
- **消融研判重点**：统计比较掺杂化学式（如 $x=0.15$、$\text{La}_{2-x}\text{Sr}_x\text{CuO}_4$）的实体召回率变化，以及由实体识别精度提升带来的下游 `REL Strict F1` 的绝对百分点提升。

---

## 6. 一键运行脚本 (`run_supermat.ps1`) 模板

在 `experiments/pure` 目录下保存为 `run_supermat.ps1`，方便一键复现或回归测试：

```powershell
# run_supermat.ps1
# PURE + MatSciBERT 在 SuperMat 数据集上的端到端一键执行脚本

$ErrorActionPreference = "Stop"
Write-Host "=== 正在启动 MatSciBERT + PURE (SuperMat) 全流水线 ===" -ForegroundColor Cyan

# 1. 激活虚拟环境
$VENV_ACTIVATE = ".\venv\Scripts\Activate.ps1"
if (Test-Path $VENV_ACTIVATE) {
    Write-Host "[1/5] 激活专用虚拟环境..." -ForegroundColor Green
    & $VENV_ACTIVATE
} else {
    Write-Error "未检测到 venv 虚拟环境，请先初始化环境！"
}

# 2. 检查数据路径
$DATA_DIR = "materials_data/json"
if (-not (Test-Path "$DATA_DIR/train.json")) {
    Write-Error "未找到 $DATA_DIR/train.json，请先执行数据转换脚本 scripts/convert_supermat_to_pure.py！"
}

# 3. 第一阶段：实体识别微调
$ENT_OUT = "materials_models/ent-matscib-ctx300"
Write-Host "[2/5] 开始微调 Stage 1 实体识别模型 (MatSciBERT)..." -ForegroundColor Green
python run_entity.py `
  --do_train --do_eval --eval_test `
  --task supermat `
  --data_dir $DATA_DIR `
  --model m3rg-iitd/matscibert `
  --output_dir $ENT_OUT `
  --learning_rate 1e-5 `
  --task_learning_rate 5e-4 `
  --train_batch_size 16 `
  --context_window 300 `
  --max_span_length 10

# 4. 第二阶段：关系抽取微调
$REL_OUT = "materials_models/rel-matscib-ctx100"
Write-Host "[3/5] 开始微调 Stage 2 关系抽取模型 (Typed Markers)..." -ForegroundColor Green
python run_relation.py `
  --do_train --do_eval --eval_test `
  --task supermat `
  --data_dir $DATA_DIR `
  --model m3rg-iitd/matscibert `
  --do_lower_case `
  --entity_output_dir $ENT_OUT `
  --output_dir $REL_OUT `
  --learning_rate 2e-5 `
  --train_batch_size 32 `
  --max_seq_length 256 `
  --context_window 100

# 5. 第三阶段：端到端严格对齐评估
Write-Host "[4/5] 执行端到端 Strict F1 评估..." -ForegroundColor Green
python run_eval.py --prediction_file "$REL_OUT/predictions.json"

Write-Host "=== MatSciBERT + PURE 流水线全流程运行完成！成果保存在 $REL_OUT ===" -ForegroundColor Cyan
```
