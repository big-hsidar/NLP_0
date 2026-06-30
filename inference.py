"""
单句联想演示：输入前缀，输出 Top-K 推荐字
"""
from models.lstm import CharLSTM
import config
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_lstm():
    inference_path = os.path.join("models", "lstm_inference.pt")
    training_path = os.path.join("models", "best_lstm.pt")

    if os.path.exists(inference_path):
        lstm_path = inference_path
    elif os.path.exists(training_path):
        lstm_path = training_path
    else:
        print(f"错误：未找到 LSTM 模型.")
        sys.exit(1)

    checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocab"]
    model = CharLSTM(len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def main():
    print("输入法联想演示.")
    print("加载模型...")
    model, vocab = load_lstm()
    print(f"词表大小: {len(vocab)}")
    print(f"输入前缀，模型返回 Top-{config.NUM_SUGGESTIONS} 候选字")
    print("输入 'q' 退出\n")

    while True:
        prompt = input("请输入前缀: ").strip()
        if prompt.lower() == "q":
            print("再见！")
            break
        if not prompt:
            continue

        suggestions = model.suggest(prompt, vocab, k=config.NUM_SUGGESTIONS)
        print("\nTop-{} 候选字:".format(config.NUM_SUGGESTIONS))
        for i, (char, prob) in enumerate(suggestions, 1):
            bar = "█" * int(prob * 50)
            print(f"  {i}. '{char}'  ({prob:.4f}) {bar}")
        print()


if __name__ == "__main__":
    main()
