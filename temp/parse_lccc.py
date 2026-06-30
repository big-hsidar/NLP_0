"""
解析 LCCC-base_train.json → 去空格 → 去重 → 采样 100k → 80/20 切分
输出: data/raw/lccc_train.txt 和 data/raw/lccc_test.txt
"""
import json
import os
import random
import re
from pathlib import Path

# ---- 路径配置 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SENTENCE_DIR = RAW_DIR / ".pending-1783318075-LCCC-base-split"
JSON_INPUT = SENTENCE_DIR / "LCCC-base_train.json"
TRAIN_OUTPUT = RAW_DIR / "lccc_train.txt"
TEST_OUTPUT = RAW_DIR / "lccc_test.txt"

TARGET_SAMPLES = 100_000
TRAIN_RATIO = 0.8
SEED = 42

# ---- 主流程 ----


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # 1. 加载 JSON 并提取所有 utterance
    print(f"读取 {JSON_INPUT} ...")
    with open(JSON_INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"对话数: {len(data)}")

    utterances = []
    for conversation in data:
        for utt in conversation:
            # 去空格: "啊 我 好 爱" → "啊我好爱"
            # 也移除多余空白
            utt = re.sub(r"\s+", "", utt)
            if utt:
                utterances.append(utt)

    print(f"总 utterance 数: {len(utterances)}")

    # 2. 去重
    before_dedup = len(utterances)
    utterances = list(dict.fromkeys(utterances))  # 保持顺序去重
    print(f"去重: {before_dedup} → {len(utterances)}")

    # 3. 随机采样
    random.seed(SEED)
    if len(utterances) > TARGET_SAMPLES:
        utterances = random.sample(utterances, TARGET_SAMPLES)
        print(f"采样: {TARGET_SAMPLES} 条")
    else:
        print(f"不足 {TARGET_SAMPLES} 条，全部保留")

    # 4. 打乱
    random.shuffle(utterances)

    # 5. 80/20 切分
    split_idx = int(len(utterances) * TRAIN_RATIO)
    train = utterances[:split_idx]
    test = utterances[split_idx:]

    print(f"训练集: {len(train)} 条")
    print(f"测试集: {len(test)} 条")

    # 6. 写入文件
    for path, data_list in [(TRAIN_OUTPUT, train), (TEST_OUTPUT, test)]:
        with open(path, "w", encoding="utf-8") as f:
            for line in data_list:
                f.write(line + "\n")
        print(f"已写入: {path}")

    print("完成！")


if __name__ == "__main__":
    main()
