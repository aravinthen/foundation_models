import torch

from llms.BabyGPT import LayerNorm


def test_layernorm_last_dimension():
    # check that the layer norm operates independently on each token

    torch.manual_seed(42)

    batch_size = 3
    seq_len = 5
    emb_dim = 8

    x = torch.randn(batch_size, seq_len, emb_dim)

    norm = LayerNorm(emb_dim)
    output = norm(x)

    mean = output.mean(dim=-1)
    var = output.var(dim=-1, unbiased=False)

    assert torch.allclose(
        mean,
        torch.zeros_like(mean),
        atol=1e-6,
    )

    assert torch.allclose(
        var,
        torch.ones_like(var),
        atol=1e-4,
    )

def test_layernorm_preserves_shape():
    # test if layer normalization preserves the shape of the batch

    x = torch.randn(2, 5, 8)

    norm = LayerNorm(8)
    output = norm(x)

    assert output.shape == x.shape

from llms.BabyGPT import TransformerBlock


def test_transformer_preserves_shape():
    # check that the transformer block preserves shape

    torch.manual_seed(42)

    config = {
        "vocab_size": 100,
        "context_length": 16,
        "emb_dim": 12,
        "n_heads": 3,
        "n_layers": 2,
        "drop_rate": 0.0,
    }

    block = TransformerBlock(config)

    x = torch.randn(
        2,                  # batch
        5,                  # sequence
        config["emb_dim"],  # embedding
    )

    output = block(x)

    assert output.shape == x.shape