"""
超参数配置文件 —— 4层 LSTM + Bigram 对比实验
"""

# ---- 数据 ----
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TRAIN_FILE = "data/raw/lccc_train.txt"
TEST_FILE = "data/raw/lccc_test.txt"

# ---- 滑动窗口 ----
SEQ_LEN = 30          # 输入前缀长度（字符数）
STRIDE = 1            # 窗口滑动步长

# ---- 词表 ----
MIN_FREQ = 2          # 最低词频阈值（低于此频次的字替换为 <UNK>）

# ---- 模型 (4层 LSTM) ----
EMBED_DIM = 512       # 字嵌入维度
HIDDEN_DIM = 1024     # LSTM 隐藏层维度
NUM_LAYERS = 4        # LSTM 层数
DROPOUT = 0.5         # 层间 Dropout（抑制过拟合）

# ---- 训练 ----
BATCH_SIZE = 256
LR = 5e-4             # 初始学习率（更平滑收敛）
WEIGHT_DECAY = 1e-4   # L2 正则（抑制过拟合）
EPOCHS = 50
LR_PATIENCE = 6       # ReduceLROnPlateau 耐心值
LR_FACTOR = 0.5       # 学习率衰减因子
EARLY_STOP_PATIENCE = 12

# ---- 评估 ----
TOP_K_LIST = [1, 3, 5, 10]

# ---- 推理 ----
NUM_SUGGESTIONS = 5   # 联想时返回的候选字数量

# ---- 设备 ----
DEVICE = "cuda"       # "cuda" / "cpu"
