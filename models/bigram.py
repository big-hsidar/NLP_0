"""
Bigram 统计语言模型基线（Laplace smoothing）
"""
import torch
import os
from collections import Counter, defaultdict
import config


class BigramModel:
    """Bigram (1-gram context) 字符级语言模型"""

    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        # 条件概率表: prob_table[prev_idx][next_idx] = P(next | prev)
        self.prob_table = None

    def train(self, filepath: str):
        """从文本文件统计 Bigram 频率，计算 Laplace 平滑概率"""
        pair_counts = defaultdict(Counter)
        unigram_counts = Counter()

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                indices = self.vocab.encode(line)
                for i in range(len(indices) - 1):
                    prev, nxt = indices[i], indices[i + 1]
                    pair_counts[prev][nxt] += 1
                    unigram_counts[prev] += 1

        # 构建概率表 (Laplace smoothing)
        self.prob_table = torch.zeros(self.vocab_size, self.vocab_size)

        for prev_idx in range(self.vocab_size):
            prev_count = unigram_counts.get(prev_idx, 0)
            for next_idx in range(self.vocab_size):
                count = pair_counts[prev_idx].get(next_idx, 0)
                # Laplace: (count + 1) / (prev_count + vocab_size)
                self.prob_table[prev_idx][next_idx] = (count + 1) / (
                    prev_count + self.vocab_size
                )

        print(f"Bigram 训练完成，词表大小: {self.vocab_size}")

    def suggest(self, context: str, k: int = None) -> list[tuple[str, float]]:
        """
        输入法联想接口：基于最后一个字符预测下一个

        Returns:
            [(char, probability), ...] Top-K
        """
        if k is None:
            k = config.NUM_SUGGESTIONS
        if self.prob_table is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        # 取最后一个字符作为上下文
        if not context:
            # 空上下文：使用均匀分布（实际上退回 unigram）
            probs = torch.ones(self.vocab_size) / self.vocab_size
        else:
            last_char = context[-1]
            prev_idx = self.vocab.char2idx.get(last_char, self.vocab.unk_idx)
            probs = self.prob_table[prev_idx]

        # 过滤特殊 token
        probs[0] = 0  # <PAD>
        probs[1] = 0  # <UNK>

        topk_values, topk_indices = torch.topk(probs, k)

        results = []
        for idx, prob in zip(topk_indices.tolist(), topk_values.tolist()):
            char = self.vocab.idx2char.get(idx, "<UNK>")
            results.append((char, prob))

        return results

    def batch_predict(self, prefixes: list[str]) -> torch.Tensor:
        """
        批量预测：给定多个前缀字符串，返回每个前缀的完整概率分布

        Args:
            prefixes: list of string, 每个元素是前缀文本

        Returns:
            probs: (batch_size, vocab_size)
        """
        batch_size = len(prefixes)
        probs = torch.zeros(batch_size, self.vocab_size)

        for i, prefix in enumerate(prefixes):
            if not prefix:
                probs[i] = torch.ones(self.vocab_size) / self.vocab_size
            else:
                last_char = prefix[-1]
                prev_idx = self.vocab.char2idx.get(
                    last_char, self.vocab.unk_idx)
                probs[i] = self.prob_table[prev_idx]

        return probs

    def save(self, filepath: str):
        torch.save(
            {
                "prob_table": self.prob_table,
                "vocab": self.vocab,
            },
            filepath,
        )
        print(f"Bigram 模型已保存: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        checkpoint = torch.load(filepath, weights_only=False)
        instance = cls(checkpoint["vocab"])
        instance.prob_table = checkpoint["prob_table"]
        instance.vocab_size = len(instance.vocab)
        print(f"Bigram 模型已加载: {filepath}")
        return instance


def build_bigram(vocab) -> BigramModel:
    """
    构建/加载 Bigram 模型
    """
    bigram_path = os.path.join(config.PROCESSED_DIR, "bigram.pt")

    if os.path.exists(bigram_path):
        return BigramModel.load(bigram_path)

    print("训练 Bigram 模型...")
    model = BigramModel(vocab)
    model.train(config.TRAIN_FILE)
    model.save(bigram_path)
    return model
