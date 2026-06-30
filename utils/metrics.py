"""
评估指标：Top-K 准确率、MRR (Mean Reciprocal Rank)
"""
import torch


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """
    计算 Top-K 准确率

    Args:
        logits: (batch_size, vocab_size) 模型输出的 logits
        targets: (batch_size,) 真实标签索引
        k: Top-K 的 K

    Returns:
        accuracy: float
    """
    _, topk_indices = torch.topk(logits, k, dim=-1)
    correct = (topk_indices == targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def mean_reciprocal_rank(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算 MRR

    Args:
        logits: (batch_size, vocab_size)
        targets: (batch_size,)

    Returns:
        mrr: float
    """
    # 按概率降序排列
    _, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    batch_size = logits.size(0)

    ranks = []
    for i in range(batch_size):
        target_positions = (
            sorted_indices[i] == targets[i]).nonzero(as_tuple=True)
        if len(target_positions[0]) > 0:
            rank = target_positions[0][0].item() + 1  # 1-indexed
            ranks.append(1.0 / rank)
        else:
            ranks.append(0.0)

    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)


def evaluate_model(model, dataloader, device: str, top_k_list: list[int]):
    """
    批量评估模型

    Returns:
        topk_accs: dict[int, float]  e.g. {1: 0.35, 3: 0.52, ...}
        mrr: float
        total_loss: float
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    # 累积每个 K 的正确数
    correct_counts = {k: 0 for k in top_k_list}
    mrr_sum = 0.0

    with torch.no_grad():
        for padded, targets, lengths in dataloader:
            padded = padded.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            logits = model(padded, lengths)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for k in top_k_list:
                correct_counts[k] += top_k_accuracy(
                    logits, targets, k) * batch_size

            mrr_sum += mean_reciprocal_rank(logits, targets) * batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    topk_accs = {k: correct_counts[k] / total_samples for k in top_k_list}
    mrr = mrr_sum / total_samples if total_samples > 0 else 0.0

    return topk_accs, mrr, avg_loss
