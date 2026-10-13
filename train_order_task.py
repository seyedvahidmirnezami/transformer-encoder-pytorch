"""Why positional encoding matters, in one experiment.

Task: each sequence contains exactly one "A" and one "B" among random filler
tokens. Label = 1 if A comes before B, else 0. The words are identical in both
classes; only their ORDER differs.

Self-attention alone is order-blind: shuffle the tokens and the [CLS] output
doesn't change. So an encoder without positional encoding can't do better than
chance here, while the same encoder with positional encoding learns the task.

Runs on a CPU in under a minute:  python train_order_task.py
"""

import torch
import torch.nn as nn

from encoder import EncoderClassifier

PAD, CLS, A, B = 0, 1, 2, 3
FILLER = range(4, 20)
VOCAB = 20
SEQ_LEN = 10  # tokens after [CLS]


def make_batch(batch_size: int, generator: torch.Generator):
    tokens = torch.randint(min(FILLER), max(FILLER) + 1, (batch_size, SEQ_LEN), generator=generator)
    labels = torch.randint(0, 2, (batch_size,), generator=generator)
    for row in range(batch_size):
        i, j = torch.randperm(SEQ_LEN, generator=generator)[:2].tolist()
        first, second = min(i, j), max(i, j)
        a_pos, b_pos = (first, second) if labels[row] == 1 else (second, first)
        tokens[row, a_pos], tokens[row, b_pos] = A, B
    cls = torch.full((batch_size, 1), CLS)
    return torch.cat([cls, tokens], dim=1), labels


def train(use_positional_encoding: bool, steps: int = 600, seed: int = 0) -> float:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = EncoderClassifier(
        num_classes=2,
        vocab_size=VOCAB,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        max_len=SEQ_LEN + 1,
        dropout=0.0,
        use_positional_encoding=use_positional_encoding,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for step in range(steps):
        x, y = make_batch(64, gen)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        x, y = make_batch(2000, torch.Generator().manual_seed(seed + 1))
        accuracy = (model(x).argmax(-1) == y).float().mean().item()
    return accuracy


if __name__ == "__main__":
    for use_pe in (False, True):
        acc = train(use_positional_encoding=use_pe)
        label = "with positional encoding   " if use_pe else "without positional encoding"
        print(f"{label}: test accuracy = {acc:.1%}")
