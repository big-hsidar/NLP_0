# 输入法联想：基于 4 层 LSTM 与 N-gram 的对比研究

> NLP · 文本生成/语言模型方向

本项目实现了一个基于 **4 层 LSTM** 的中文输入法联想引擎，并与传统的 **Bigram 统计语言模型**进行系统对比。

- **数据**：LCCC-base 对话语料，采样 100k 条 utterance，80k 训练 / 20k 测试
- **模型**：4 层 LSTM（~37M 参数，Embedding 512 / Hidden 1024）
- **结果**：LSTM Top-1 准确率 **16.27%**，Bigram Top-1 仅 3.03%；MRR 0.24 vs 0.07，**神经网络全面碾压 N-gram**

## 🧠 核心思想

输入法联想本质是一个**因果语言模型**（Causal LM）：给定用户已输入的前缀文本，预测下一个最可能的字。  
传统输入法引擎（如 fcitx5 默认引擎）多基于 N-gram 统计，仅看前 1~2 个词，缺乏语义理解能力。本项目采用 4 层 LSTM 捕捉长距离依赖，并在真实对话短文本上验证其效果。

## 📁 项目结构

```
Main/
├── data/
│   ├── raw/                      # 原始语料（lccc_train.txt, lccc_test.txt）
│   └── processed/                # .pt 序列化数据（train.pt, test.pt, vocab.pt, bigram.pt）
├── models/
│   ├── lstm.py                   # 4层 LSTM 模型定义（含 suggest() 联想接口）
│   └── bigram.py                 # Bigram 基线模型（Laplace 平滑）
├── utils/
│   ├── data_loader.py            # 词表构建、滑动窗口生成、DataLoader
│   └── metrics.py                # Top-K 准确率、MRR 评估
├── train_lstm.py                 # LSTM 训练脚本（早停 + 学习率衰减 + label smoothing）
├── train_bigram.py               # Bigram 训练脚本（频次统计 → Laplace 概率表）
├── evaluate.py                   # LSTM vs Bigram 对比评估
├── inference.py                  # 交互式联想 Demo
├── config.py                     # 超参数配置
├── requirements.txt              # Python 依赖
├── temp/
│   ├── parse_lccc.py             # JSON → 去空格 → 采样 → 切分
│   └── slim_checkpoint.py        # 训练 checkpoint 瘦身为推理版本
└── readme.md                     # 本文件
```

## 📊 实验结果

| 指标 | LSTM (4-layer) | Bigram (Laplace) | LSTM / Bigram |
|------|---------------|------------------|---------------|
| Top-1 准确率 | **16.27%** | 3.03% | **5.4×** |
| Top-3 准确率 | **27.04%** | 6.40% | **4.2×** |
| Top-5 准确率 | **32.49%** | 10.13% | **3.2×** |
| Top-10 准确率 | **39.88%** | 16.28% | **2.4×** |
| MRR | **0.2432** | 0.0714 | **3.4×** |

- 词表大小：3,733 | 训练样本：61,870 | 测试样本：15,363
- LSTM Epoch 12 即达以上结果，Bigram 仅依赖前 1 个字预测，命中率极低
- **结论：神经网络语言模型在输入法联想任务上显著优于 N-gram 统计方法**

## 🚀 实验操作流程

```bash
cd Main

# 0. 安装依赖（一次性）
pip install -r requirements.txt

# 1. 训练 Bigram 基线（纯统计）
python train_bigram.py

# 2. 训练 LSTM（约 12~25 轮触发早停）
python train_lstm.py

# 3. 对比评估
python evaluate.py

# 4. 交互式联想 Demo
python inference.py
```

## 🗜️ 模型文件说明

| 文件 | 大小 | 内容 |
|------|------|------|
| `models/best_lstm.pt` | ~426 MB | 完整训练 checkpoint（模型权重 + 优化器状态 + 词表） |
| `models/lstm_inference.pt` | ~142 MB | 推理版（仅模型权重 + 词表），用于 `inference.py` |

> **注意**：模型和数据文件（`.pt`）已通过 `.gitignore` 排除，不会提交到 Git 仓库。
> 克隆后需按上方步骤运行 `train_bigram.py` → `train_lstm.py` 即可生成所有文件。

训练完成后运行 `python temp/slim_checkpoint.py` 可生成瘦身版，减小 67%。

## ⚙️ 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| SEQ_LEN | 30 | 前缀长度（字符数） |
| EMBED_DIM | 512 | 字嵌入维度 |
| HIDDEN_DIM | 1024 | LSTM 隐藏层维度 |
| NUM_LAYERS | 4 | LSTM 层数 |
| DROPOUT | 0.5 | 层间 Dropout |
| BATCH_SIZE | 256 | 批次大小 |
| LR | 5e-4 | 初始学习率 |
| WEIGHT_DECAY | 1e-4 | L2 正则 |
| LABEL_SMOOTHING | 0.1 | 标签平滑 |
| EARLY_STOP_PATIENCE | 12 | 早停耐心值 |

## 📦 依赖

```
torch>=2.0
numpy
tqdm
tabulate