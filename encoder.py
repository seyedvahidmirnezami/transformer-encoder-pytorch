"""A transformer encoder built from scratch in PyTorch.

Pieces, in the order data flows through them:

    tokens ─▶ embedding ─▶ + positional encoding ─▶ [EncoderBlock] × N ─▶ one vector per token

    EncoderBlock:
        Z   = LayerNorm(X + MultiHeadAttention(X))     communicate
        out = LayerNorm(Z + FeedForward(Z))            think

Shapes are written as (batch, n, d_model) throughout.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).

    Returns a (max_len, d_model) table. No parameters: every position gets a
    unique, bounded pattern, and the formula works for any length.
    """
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)  # 0, 2, 4, ...
    angle = pos / torch.pow(10000.0, two_i / d_model)  # (max_len, d_model/2)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with h heads, concatenated and projected by W_O."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly into heads"
        self.h, self.d_k = num_heads, d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.h, self.d_k).transpose(1, 2)  # (b, h, n, d_k)
        Q, K, V = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (b, h, n, n)
        if pad_mask is not None:  # pad_mask: (b, n), True at padding positions
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        A = self.dropout(scores.softmax(dim=-1))
        out = (A @ V).transpose(1, 2).reshape(b, n, d)
        return self.W_O(out), A


class FeedForward(nn.Module):
    """FFN(x) = ReLU(x W₁ + b₁) W₂ + b₂, applied to each token on its own.

    Expand to d_ff (usually 4 × d_model), then compress back to d_model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Multi-head attention and a feed-forward network, each wrapped in Add & Norm.

    Post-norm, as in the original paper and BERT: LayerNorm(x + SubLayer(x)).
    Output shape equals input shape, so blocks stack.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        attn_out, A = self.attn(x, pad_mask)
        z = self.norm1(x + self.drop(attn_out))  # residual: keep the original signal
        out = self.norm2(z + self.drop(self.ffn(z)))
        return out, A


class TransformerEncoder(nn.Module):
    """Embedding + positional encoding + N encoder blocks.

    Position 0 of every input is expected to be a [CLS] token; its output
    vector summarizes the whole sequence for sentence-level tasks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.use_pe = use_positional_encoding
        self.register_buffer("pe", sinusoidal_positional_encoding(max_len, d_model), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, tokens: torch.Tensor):
        """tokens: (b, n) integer ids. Returns (b, n, d_model) and per-layer attention."""
        pad_mask = tokens == self.pad_id
        x = self.embed(tokens) * math.sqrt(self.d_model)
        if self.use_pe:
            x = x + self.pe[: tokens.size(1)]  # position info is added once, here
        x = self.drop(x)
        attentions = []
        for block in self.blocks:
            x, A = block(x, pad_mask)
            attentions.append(A)
        return x, attentions


class EncoderClassifier(nn.Module):
    """Sentence classification from the [CLS] output vector."""

    def __init__(self, num_classes: int, **encoder_kwargs):
        super().__init__()
        self.encoder = TransformerEncoder(**encoder_kwargs)
        self.head = nn.Linear(self.encoder.d_model, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encoder(tokens)
        return self.head(hidden[:, 0])  # logits from the [CLS] position


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TransformerEncoder(vocab_size=100, d_model=64, num_heads=4, d_ff=256, num_layers=2)
    tokens = torch.randint(1, 100, (2, 7))
    hidden, attentions = model(tokens)
    print("tokens     ", tuple(tokens.shape))
    print("hidden     ", tuple(hidden.shape), "(one vector per token)")
    print("attention  ", len(attentions), "layers of", tuple(attentions[0].shape))
    print("parameters ", sum(p.numel() for p in model.parameters()))
