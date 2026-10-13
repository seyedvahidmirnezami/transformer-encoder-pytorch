"""Check the encoder against the course numbers and its key properties."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoder import (  # noqa: E402
    EncoderBlock,
    TransformerEncoder,
    sinusoidal_positional_encoding,
)

TOL = dict(atol=1e-3, rtol=0)


def test_positional_encoding_matches_session_6():
    pe = sinusoidal_positional_encoding(max_len=3, d_model=4)
    expected = torch.tensor(
        [
            [0.000, 1.000, 0.000, 1.000],
            [0.841, 0.540, 0.010, 1.000],
            [0.909, -0.416, 0.020, 1.000],
        ]
    )
    torch.testing.assert_close(pe, expected, **TOL)


def test_adding_positions_to_bank_of_river():
    X = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    combined = X + sinusoidal_positional_encoding(3, 4)
    expected = torch.tensor(
        [
            [1.000, 1.000, 1.000, 1.000],
            [0.841, 1.540, 0.010, 2.000],
            [1.909, 0.584, 0.020, 1.000],
        ]
    )
    torch.testing.assert_close(combined, expected, **TOL)


def test_layer_norm_matches_session_5():
    x = torch.tensor([[2.0, 1.0, 2.0, 1.436]])
    out = torch.nn.LayerNorm(4)(x)  # gamma = 1, beta = 0 at initialization
    # The lecture shows -1.450 and -0.411 because it rounds the mean and std
    # before dividing; full precision gives -1.449 and -0.412.
    torch.testing.assert_close(out, torch.tensor([[0.930, -1.449, 0.930, -0.412]]), **TOL)


def test_encoder_block_keeps_shape():
    block = EncoderBlock(d_model=16, num_heads=4, d_ff=64)
    x = torch.randn(3, 9, 16)
    out, A = block(x)
    assert out.shape == x.shape
    assert A.shape == (3, 4, 9, 9)


def _cls_output(model, tokens):
    model.eval()
    with torch.no_grad():
        hidden, _ = model(tokens)
    return hidden[:, 0]


def test_without_positions_attention_is_order_blind():
    torch.manual_seed(0)
    model = TransformerEncoder(vocab_size=30, d_model=16, num_heads=4, d_ff=32, use_positional_encoding=False)
    tokens = torch.tensor([[1, 5, 6, 7, 8, 9]])
    shuffled = torch.tensor([[1, 9, 7, 5, 8, 6]])  # same words, [CLS] still first
    torch.testing.assert_close(_cls_output(model, tokens), _cls_output(model, shuffled))


def test_with_positions_word_order_changes_the_output():
    torch.manual_seed(0)
    model = TransformerEncoder(vocab_size=30, d_model=16, num_heads=4, d_ff=32, use_positional_encoding=True)
    tokens = torch.tensor([[1, 5, 6, 7, 8, 9]])
    shuffled = torch.tensor([[1, 9, 7, 5, 8, 6]])
    assert not torch.allclose(_cls_output(model, tokens), _cls_output(model, shuffled), atol=1e-4)


def test_padding_gets_no_attention():
    torch.manual_seed(0)
    model = TransformerEncoder(vocab_size=30, d_model=16, num_heads=4, d_ff=32)
    model.eval()
    tokens = torch.tensor([[1, 5, 6, 0, 0]])  # 0 is padding
    _, attentions = model(tokens)
    assert torch.all(attentions[0][..., 3:] == 0)
