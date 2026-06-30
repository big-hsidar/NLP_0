"""
4层 LSTM 字符级语言模型（含输入法联想接口）
"""
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import config


class CharLSTM(nn.Module):
    """4层 LSTM 因果语言模型"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size, config.EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=config.EMBED_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            batch_first=True,
        )
        self.head = nn.Linear(config.HIDDEN_DIM, vocab_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) padded token indices
            lengths: (batch,) actual lengths

        Returns:
            logits: (batch, vocab_size)  仅取每个序列最后一个有效 token 的输出
        """
        batch_size, seq_len = x.shape

        # Embedding
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Pack padded sequence
        packed = rnn_utils.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_out, _ = self.lstm(packed)

        # Unpack
        lstm_out, _ = rnn_utils.pad_packed_sequence(
            packed_out, batch_first=True)
        # lstm_out: (batch, seq_len, hidden_dim)

        # 取出每个序列最后一个有效位置的输出
        last_indices = (lengths - 1).long().unsqueeze(-1).unsqueeze(-1)
        last_indices = last_indices.expand(-1, -1, lstm_out.size(-1))
        last_out = lstm_out.gather(1, last_indices).squeeze(1)
        # last_out: (batch, hidden_dim)

        logits = self.head(last_out)  # (batch, vocab_size)
        return logits

    def suggest(self, prompt: str, vocab, k: int = None) -> list[tuple[str, float]]:
        """
        输入法联想接口

        Args:
            prompt: 用户输入的前缀字符串
            vocab: CharVocab 实例
            k: 返回 Top-K 候选字

        Returns:
            [(char, probability), ...] 按概率降序
        """
        if k is None:
            k = config.NUM_SUGGESTIONS
        self.eval()

        with torch.no_grad():
            # 取最后 SEQ_LEN 个字符
            if len(prompt) > config.SEQ_LEN:
                prompt = prompt[-config.SEQ_LEN:]

            indices = vocab.encode(prompt)
            x = torch.tensor([indices], dtype=torch.long)
            lengths = torch.tensor([len(indices)], dtype=torch.long)

            # 直接前向传播（不用 padding）
            emb = self.embedding(x)
            lstm_out, _ = self.lstm(emb)
            last_out = lstm_out[:, -1, :]  # 取最后一个时间步
            logits = self.head(last_out)  # (1, vocab_size)

            probs = torch.softmax(logits, dim=-1).squeeze(0)

            # 过滤特殊 token
            probs[0] = 0  # <PAD>
            probs[1] = 0  # <UNK>

            topk_values, topk_indices = torch.topk(probs, k)

            results = []
            for idx, prob in zip(topk_indices.tolist(), topk_values.tolist()):
                char = vocab.idx2char.get(idx, "<UNK>")
                results.append((char, prob))

        return results
