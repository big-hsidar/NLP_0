"""
LSTM 模型训练脚本（含早停、学习率衰减）
"""
from models.lstm import CharLSTM
from utils.metrics import evaluate_model
from utils.data_loader import build_and_save, get_dataloaders
import config
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Train")
    for padded, targets, lengths in pbar:
        padded = padded.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(padded, lengths)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / total_samples


def train():
    print("准备数据...")
    vocab, train_inputs, train_targets, test_inputs, test_targets = build_and_save()
    train_loader, test_loader = get_dataloaders(
        vocab, train_inputs, train_targets, test_inputs, test_targets
    )
    vocab_size = len(vocab)
    print(f"词表大小: {vocab_size}")
    print(f"训练样本数: {len(train_inputs)}")
    print(f"测试样本数: {len(test_inputs)}")

    print("\n构建模型...")
    device = torch.device(
        config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model = CharLSTM(vocab_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,} (~{total_params // 1_000_000}M)")
    print(f"可训练参数量: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx, label_smoothing=0.1)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        verbose=True,
    )

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_lstm.pt")

    best_val_loss = float("inf")
    patience_counter = 0

    print("\n开始训练...")

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device)
        print(f"训练 Loss: {train_loss:.4f}")

        topk_accs, mrr, val_loss = evaluate_model(
            model, test_loader, device, config.TOP_K_LIST
        )
        print(f"验证 Loss: {val_loss:.4f}")
        for k, acc in topk_accs.items():
            print(f"  Top-{k} 准确率: {acc:.4f}")
        print(f"  MRR: {mrr:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab": vocab,
                    "val_loss": val_loss,
                    "topk_accs": topk_accs,
                    "mrr": mrr,
                },
                best_model_path,
            )
            print(f"  [最佳模型已保存] val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            print(
                f"  未提升 (patience={patience_counter}/{config.EARLY_STOP_PATIENCE})"
            )
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print("早停触发！")
                break

    print(f"\n训练完成！最佳验证 Loss: {best_val_loss:.4f}")
    print(f"模型已保存至: {best_model_path}")


if __name__ == "__main__":
    train()
