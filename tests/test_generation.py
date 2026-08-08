import torch
from llms.BabyGPT import BabyGPTModel, generate_text

def test_sampled_generation_batch_tokens():
    """
    Tests that the sampled generation preserves batch dimension and correctly adds tokens.
    :return:
    """
    torch.manual_seed(42)

    config = {
        "vocab_size": 100,
        "context_length": 16,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0
    }

    model = BabyGPTModel(config)
    model.eval()

    # two independent sequences, each initially four tokens long
    idx = torch.tensor([[1, 2, 3, 4],
                        [5, 6, 7, 8]])

    max_new_tokens = 3
    output = generate_text(model=model,
                           idx = idx,
                           max_new_tokens=max_new_tokens,
                           context_size=config["context_length"],
                           mode="sampler",
                           temperature=1.0)

    # check that the batching and output is correct
    assert output.shape == (2, 4+max_new_tokens)
    assert output.dtype == torch.long
    assert torch.equal(output[:, :4], idx)
