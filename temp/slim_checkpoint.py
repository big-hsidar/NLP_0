"""
将完整训练 checkpoint 瘦身为纯推理版本（仅保留模型权重 + 词表）
用法: python temp/slim_checkpoint.py [input.pt] [output.pt]
默认: models/best_lstm.pt → models/lstm_inference.pt
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))


def slim(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"错误：未找到输入文件 {input_path}")
        sys.exit(1)

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    required = ["model_state_dict", "vocab"]
    missing = [k for k in required if k not in checkpoint]
    if missing:
        print(f"错误：checkpoint 缺少必要字段: {missing}")
        sys.exit(1)

    slim_ckpt = {
        "model_state_dict": checkpoint["model_state_dict"],
        "vocab": checkpoint["vocab"],
    }

    torch.save(slim_ckpt, output_path)

    input_size = os.path.getsize(input_path) / 1024 / 1024
    output_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"瘦身完成:")
    print(f"  {input_path}: {input_size:.1f} MB")
    print(f"  {output_path}: {output_size:.1f} MB")
    print(f"  缩减: {(1 - output_size / input_size) * 100:.0f}%")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(
        sys.argv) > 1 else os.path.join("models", "best_lstm.pt")
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        "models", "lstm_inference.pt")
    slim(input_path, output_path)
