# Transformer Encoder from Scratch in PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seyedvahidmirnezami/transformer-encoder-pytorch/blob/main/transformer_encoder_from_scratch.ipynb)
[![Course on Udemy](https://img.shields.io/badge/Full%20course-Udemy-A435F0?logo=udemy&logoColor=white)](https://www.udemy.com/course/master-transformers-from-scratch-bert-vit-ai-search/)

The transformer encoder, the building block of BERT, built piece by piece in about 150 lines of readable PyTorch:
positional encoding, multi-head attention, Add & Norm, the feed-forward network, and stacked encoder blocks.

Includes a small experiment that shows **why positional encoding exists**: the same model scores 50% (a coin flip) without it and 100% with it.

## The architecture

```
tokens ─▶ embedding ─▶ + positional encoding ─▶ [ encoder block ] × N ─▶ one vector per token
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         │  Z   = LayerNorm(X + MultiHeadAttention(X))   │  communicate
                         │  out = LayerNorm(Z + FeedForward(Z))          │  think
                         └───────────────────────────────────────────────┘
```

| Piece | Role |
|---|---|
| Multi-head attention | **Communication:** tokens share information |
| Feed-forward network | **Thinking:** each token is processed on its own (`d_model → 4·d_model → d_model`) |
| Residual connection | **Safety net:** keeps the original signal and gives gradients a highway |
| Layer normalization | **Stabilizer:** keeps each token's numbers well-behaved |
| Positional encoding | **Word order:** attention alone can't tell "dog bites man" from "man bites dog" |

## Positional encoding with real numbers

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

With `d_model = 4`, added to the "bank of river" embeddings:

```
           embedding        positional encoding             sum
bank  @0   [1 0 1 0]   +   [0.000  1.000 0.000 1.000]   =   [1.000 1.000 1.000 1.000]
of    @1   [0 1 0 1]   +   [0.841  0.540 0.010 1.000]   =   [0.841 1.540 0.010 2.000]
river @2   [1 1 0 0]   +   [0.909 -0.416 0.020 1.000]   =   [1.909 0.584 0.020 1.000]
```

## The experiment: does A come before B?

Every sequence hides one `A` and one `B` among random filler tokens. The label is whether `A` comes first.
Both classes contain the same words, so the only signal is **order**.

```
$ python train_order_task.py
without positional encoding: test accuracy = 50.1%
with positional encoding   : test accuracy = 100.0%
```

Without positions, shuffling the input doesn't change the `[CLS]` output at all, so the model can only guess.
Runs on a CPU in under 10 seconds. Results are the same across seeds (about 50% vs 98.5–100%).

## What's in this repo

| File | What it is |
|---|---|
| [`transformer_encoder_from_scratch.ipynb`](transformer_encoder_from_scratch.ipynb) | Step-by-step walkthrough with the course numbers, a positional-encoding heatmap and the experiment. Open it in Colab |
| [`encoder.py`](encoder.py) | `sinusoidal_positional_encoding`, `MultiHeadAttention`, `FeedForward`, `EncoderBlock`, `TransformerEncoder`, `EncoderClassifier` |
| [`train_order_task.py`](train_order_task.py) | The positional-encoding experiment |
| [`tests/test_encoder.py`](tests/test_encoder.py) | Checks the course numbers, shapes, padding masks, and that the model is order-blind without positions |

## Run it locally

```bash
pip install -r requirements.txt
python encoder.py              # shapes and parameter count
python train_order_task.py     # the experiment
pytest                         # the tests
```

## Learn the full transformer

This repo covers Sessions 5–6 of **[Master Transformers from Scratch: BERT, ViT & AI Search](https://www.udemy.com/course/master-transformers-from-scratch-bert-vit-ai-search/)**.
The course goes on to pretrain BERT from this encoder, fine-tune it for classification and NER,
adapt it to images with Vision Transformers, and build two-tower semantic search.

More from the course:
- [self-attention-from-scratch](https://github.com/seyedvahidmirnezami/self-attention-from-scratch): attention with real numbers, in NumPy and PyTorch
- [vision-transformer-cassava](https://github.com/seyedvahidmirnezami/vision-transformer-cassava): fine-tune a Vision Transformer on Kaggle
- [semantic-search-two-tower](https://github.com/seyedvahidmirnezami/semantic-search-two-tower): build two-tower retrieval from scratch

If this helped you, a ⭐ helps other learners find it.

## License

[MIT](LICENSE)
