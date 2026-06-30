"""
LSTM vs Bigram 对比评估脚本
"""
from models.bigram import BigramModel
from models.lstm import CharLSTM
from utils.metrics import evaluate_model
from utils.data_loader import build_and_save, get_dataloaders
import config
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def evaluate_bigram_on_test(bigram, test_loader, device, top_k_list):
    """在测试集上评估 Bigram（用最后一个字符做上下文）"""
    correct_counts = {k: 0 for k in top_k_list}
    mrr_sum = 0.0
    total_samples = 0
    pad_idx = bigram.vocab.pad_idx

    with torch.no_grad():
        for padded, targets, lengths in test_loader:
            padded = padded.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            batch_size = padded.size(0)

            for i in range(batch_size):
                seq_len = lengths[i].item()
                if seq_len < 2:
                    continue
                prev_idx = padded[i, seq_len - 2].item()
                target_idx = targets[i].item()

                probs = bigram.prob_table[prev_idx].clone()
                probs[pad_idx] = 0

                _, topk_indices = torch.topk(probs, max(top_k_list), dim=-1)
                for k in top_k_list:
                    if target_idx in topk_indices[:k]:
                        correct_counts[k] += 1

                sorted_indices = torch.argsort(probs, descending=True)
                rank = (sorted_indices == target_idx).nonzero(as_tuple=True)
                if len(rank[0]) > 0:
                    mrr_sum += 1.0 / (rank[0][0].item() + 1)

                total_samples += 1

    topk_accs = {
        k: correct_counts[k] / total_samples for k in top_k_list
    } if total_samples > 0 else {k: 0 for k in top_k_list}
    mrr = mrr_sum / total_samples if total_samples > 0 else 0.0
    return topk_accs, mrr


def evaluate():
    print("准备数据...")
    vocab, train_inputs, train_targets, test_inputs, test_targets = build_and_save()
    train_loader, test_loader = get_dataloaders(
        vocab, train_inputs, train_targets, test_inputs, test_targets
    )
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu")

    # ---- LSTM ----
    print("\n加载 LSTM 模型...")
    lstm_path = os.path.join("models", "best_lstm.pt")
    if not os.path.exists(lstm_path):
        print(f"错误：未找到 LSTM 模型 {lstm_path}，请先运行 train_lstm.py")
        sys.exit(1)

    lstm = CharLSTM(len(vocab)).to(device)
    checkpoint = torch.load(lstm_path, map_location=device, weights_only=False)
    lstm.load_state_dict(checkpoint["model_state_dict"])
    val_loss_val = checkpoint.get('val_loss', None)
    val_loss_str = f"{val_loss_val:.4f}" if isinstance(
        val_loss_val, float) else "N/A"
    print(f"已加载 LSTM (epoch={checkpoint.get('epoch', '?')}, "
          f"val_loss={val_loss_str})")

    lstm_topk, lstm_mrr, lstm_loss = evaluate_model(
        lstm, test_loader, device, config.TOP_K_LIST
    )

    # ---- Bigram ----
    print("\n加载 Bigram 模型...")
    bigram_path = os.path.join(config.PROCESSED_DIR, "bigram.pt")
    if not os.path.exists(bigram_path):
        print(f"错误：未找到 Bigram 模型 {bigram_path}，请先运行 train_bigram.py")
        sys.exit(1)
    bigram = BigramModel.load(bigram_path)
    bigram_topk, bigram_mrr = evaluate_bigram_on_test(
        bigram, test_loader, device, config.TOP_K_LIST
    )

    # ---- 对比表格 ----
    print("\nLSTM vs Bigram 对比结果")
    header = f"{'指标':<16} {'LSTM (4-layer)':<18} {'Bigram (Laplace)':<18}"
    print(header)

    rows = [
        ("Top-1 准确率", f"{lstm_topk[1]:.4f}", f"{bigram_topk[1]:.4f}"),
    ]
    for k in config.TOP_K_LIST[1:]:
        rows.append(
            (f"Top-{k} 准确率", f"{lstm_topk[k]:.4f}", f"{bigram_topk[k]:.4f}"))
    rows.append(("MRR", f"{lstm_mrr:.4f}", f"{bigram_mrr:.4f}"))
    rows.append(("Test Loss", f"{lstm_loss:.4f}", "N/A"))

    for metric, lstm_val, bigram_val in rows:
        print(f"{metric:<16} {lstm_val:<18} {bigram_val:<18}")


if __name__ == "__main__":
    evaluate()
