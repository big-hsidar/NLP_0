"""
Bigram 统计语言模型训练脚本（频次统计 + Laplace 平滑）
"""
from models.bigram import BigramModel
from utils.data_loader import build_and_save
import config
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def train():
    print("准备数据...")
    vocab, _, _, _, _ = build_and_save()
    print(f"词表大小: {len(vocab)}")

    bigram_path = os.path.join(config.PROCESSED_DIR, "bigram.pt")

    print("统计 Bigram 频次...")
    model = BigramModel(vocab)
    model.train(config.TRAIN_FILE)

    model.save(bigram_path)

    # 统计概率表大小
    non_zero = (model.prob_table > 0).sum().item()
    total = model.prob_table.numel()
    sparsity = 1.0 - non_zero / total
    print(
        f"概率表大小: {model.prob_table.shape} ({model.prob_table.numel() * 4 / 1024 / 1024:.1f} MB)")
    print(f"稀疏度: {sparsity:.2%}")


if __name__ == "__main__":
    train()
