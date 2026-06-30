"""
数据读取、词表构建、滑动窗口生成、.pt 序列化
"""
import os
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import config


class CharVocab:
    """字符级词表"""

    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2char = {0: "<PAD>", 1: "<UNK>"}
        self.pad_idx = 0
        self.unk_idx = 1

    def build_from_file(self, filepath: str, min_freq: int = 2):
        """从文本文件构建词表"""
        counter = Counter()
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    counter.update(line)

        # 按频率过滤并排序
        for char, freq in counter.most_common():
            if freq >= min_freq:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

        print(f"词表大小: {len(self.char2idx)} (min_freq={min_freq})")

    def __len__(self):
        return len(self.char2idx)

    def encode(self, text: str) -> list[int]:
        return [self.char2idx.get(c, self.unk_idx) for c in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx2char.get(i, "<UNK>") for i in indices)


def create_sequences(text: str, seq_len: int, stride: int = 1):
    """
    滑动窗口生成 (input_seq, target_char) 对
    input_seq: 长度为 seq_len 的前缀
    target_char: 前缀后下一个字符

    Returns:
        inputs: list[str]  每个元素是 seq_len 长度的字符串
        targets: list[str] 每个元素是单个字符
    """
    inputs = []
    targets = []

    if len(text) <= seq_len:
        return inputs, targets

    for i in range(0, len(text) - seq_len, stride):
        inputs.append(text[i: i + seq_len])
        targets.append(text[i + seq_len])

    return inputs, targets


class CharSequenceDataset(Dataset):
    """字符级序列数据集 (inputs/targets 已为 token indices)"""

    def __init__(self, inputs, targets, vocab):
        self.inputs = [torch.tensor(s, dtype=torch.long) for s in inputs]
        self.targets = list(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], torch.tensor(self.targets[idx], dtype=torch.long)


def collate_fn(batch):
    """批次整理：按序列长度降序 padding"""
    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)
    # padding
    max_len = lengths.max().item()
    padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
    for i, seq in enumerate(inputs):
        padded[i, : len(seq)] = seq
    targets = torch.stack(targets)
    return padded, targets, lengths


def build_and_save():
    """
    构建词表 + 处理训练/测试数据，保存 .pt 文件到 data/processed/
    如果已存在则跳过
    """
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    vocab_path = os.path.join(config.PROCESSED_DIR, "vocab.pt")
    train_pt = os.path.join(config.PROCESSED_DIR, "train.pt")
    test_pt = os.path.join(config.PROCESSED_DIR, "test.pt")

    # ---- 词表 ----
    if os.path.exists(vocab_path):
        print("加载已有词表...")
        vocab = torch.load(vocab_path, weights_only=False)
    else:
        print("构建词表...")
        vocab = CharVocab()
        vocab.build_from_file(config.TRAIN_FILE, min_freq=config.MIN_FREQ)
        torch.save(vocab, vocab_path)
        print(f"词表已保存: {vocab_path}")

    # ---- 训练集 ----
    if os.path.exists(train_pt):
        print("加载已有训练集 .pt ...")
        train_inputs, train_targets = torch.load(train_pt, weights_only=False)
    else:
        print("处理训练集...")
        train_inputs, train_targets = _process_file(config.TRAIN_FILE, vocab)
        torch.save((train_inputs, train_targets), train_pt)
        print(f"训练集已保存: {train_pt} ({len(train_inputs)} 条)")

    # ---- 测试集 ----
    if os.path.exists(test_pt):
        print("加载已有测试集 .pt ...")
        test_inputs, test_targets = torch.load(test_pt, weights_only=False)
    else:
        print("处理测试集...")
        test_inputs, test_targets = _process_file(config.TEST_FILE, vocab)
        torch.save((test_inputs, test_targets), test_pt)
        print(f"测试集已保存: {test_pt} ({len(test_inputs)} 条)")

    return vocab, train_inputs, train_targets, test_inputs, test_targets


def _process_file(filepath: str, vocab: CharVocab):
    """读取文本文件，生成滑动窗口序列（以编码后的 token index 存储）"""
    all_inputs = []
    all_targets = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inputs, targets = create_sequences(
                line, config.SEQ_LEN, config.STRIDE)
            if not inputs:
                continue
            encoded_inputs = [vocab.encode(s) for s in inputs]
            encoded_targets = [vocab.char2idx.get(
                t, vocab.unk_idx) for t in targets]
            all_inputs.extend(encoded_inputs)
            all_targets.extend(encoded_targets)

    return all_inputs, all_targets


def get_dataloaders(vocab, train_inputs, train_targets, test_inputs, test_targets):
    """创建 DataLoader"""
    train_dataset = CharSequenceDataset(train_inputs, train_targets, vocab)
    test_dataset = CharSequenceDataset(test_inputs, test_targets, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    build_and_save()
