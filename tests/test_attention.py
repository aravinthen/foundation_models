import torch

from attention.multi_head_attention import NaiveMultiHeadAttention, MultiHeadAttention
from attention.scaled_dot_product_attn import SelfAttention
from attention.masked_attention import MaskedAttention

def test_self_attention_output_shape():
    torch.manual_seed(42)

    seq_len = 5
    d_in = 8
    d_out = 4

    # random tensors test - sufficient for just assessing shape
    x = torch.randn(seq_len, d_in)

    attention = SelfAttention(d_in=d_in,
                              d_out=d_out)

    output = attention(x)

    # check the shape
    assert output.shape == (seq_len, d_out)

def test_masked_attention():
    """
    Main focus here - test whether masked attention actually prevents information leakage from the future.
    :return:
    """
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 6
    d_in = 8
    d_out = 4

    attention = MaskedAttention(d_in=d_in,
                                d_out=d_out,
                                context_length=seq_len,
                                dropout=0.0)

    attention.eval()

    x = torch.randn(batch_size, seq_len, d_in)

    # make identical copy and then heavily alter future tokens - these shouldn't be available to attention calculation
    x_modified = x.clone()
    x_modified[:, 3:, :] = torch.randn_like(x_modified[:, 3:, :]) * 123

    output_original = attention(x)
    output_modified = attention(x_modified)

    # the idea here is that if the future tokens are masked appropriately, the initial values here ought to be the same.
    assert torch.allclose(output_original[:, :3, :],
                          output_modified[:, :3, :],
                          atol=1e-6)

def test_naive_multihead_attention_concatenation():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_in = 8
    head_output_dim = 4
    num_heads = 3

    # test input (randomized)
    x = torch.randn(batch_size, seq_len, d_in)

    attention = NaiveMultiHeadAttention(d_in=d_in,
                                        d_out=head_output_dim,
                                        context_len=seq_len,
                                        dropout=0.0,
                                        num_heads=num_heads)

    output = attention(x)

    # check shape
    assert output.shape == (batch_size, seq_len, head_output_dim*num_heads)


def test_multihead_attention_concatenation():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_in = 8
    d_out = 12
    num_heads = 3

    # test input (randomized)
    x = torch.randn(batch_size, seq_len, d_in)

    attention = MultiHeadAttention(d_in=d_in,
                                   d_out=d_out,
                                   context_len=seq_len,
                                   dropout=0.0,
                                   num_heads=num_heads)

    output = attention(x)

    # check shape
    assert output.shape == (batch_size, seq_len, d_out)