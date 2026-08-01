#
# Program name: BabyGPT.py
# Description: A personal implementation of a GPT.
#

import torch
import torch.nn as nn
import tiktoken

from attention.multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        context_length = config["context_length"]
        emb_dim = config["emb_dim"]
        n_heads= config["n_heads"]
        drop_rate = config["drop_rate"]
        qkv_bias = config["qkv_bias"]

        # the multihead attention
        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_len=context_length,
            num_heads=n_heads,
            dropout=drop_rate,
        )

        # surrounding machinery
        self.ff = FeedForward(config, use_shortcut=False, depth=1)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x) #pre-layer norm
        x = self.att(x)
        x = self.drop(x)

        # add the original input back - improves gradient flow
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + shortcut

        return x

class LayerNorm(nn.Module):
    """
    Layer norm adjusts the outputs of a single layer to have a mean of zero and a variance of 1.
    This should be applied before and after the multi-head attention module.
    """
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps

        # trainable parameters that adjust to learn appropriate scaling and shifting
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        Simply subtracts the mean and divides by the standard deviation.
        :param x:
        :return:
        """
        mean = x.mean(dim=-1)
        var = x.var(dim=-1, keep_dim=True, unbiased=False)

        # use eps to prevent Nans
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift

class FeedForward(nn.Module):
    """
    Receives input batches. Internally expands the embedding dimension into a higher space through the first linear
    layer.
    """
    def __init__(self, config, use_shortcut, depth=2):
        super().__init__()

        emb_dim = config["emb_dim"]
        self.use_shortcut = use_shortcut

        modules = []
        for _ in range(depth):
            modules.append(nn.Linear(emb_dim, emb_dim))
            modules.append(nn.GELU())

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

# the full GPT model
class BabyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        vocab_size = config["vocab_size"]
        context_length = config["context_length"]
        emb_dim = config["emb_dim"]
        n_layers = config["n_layers"]
        drop_rate = config["drop_rate"]

        # embeddings and dropout
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(drop_rate)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config)
              for _ in range(n_layers)])

        # layer normalisation
        self.final_norm = LayerNorm(emb_dim)

        # final output in logits
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len, = in_idx.shape
        tok_embs = self.tok_emb(in_idx)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # add the positional embeddings to the token embedding
        x = tok_embs + pos_embs

        # carry out dropout
        x = self.drop_emb(x)

        # pass through the transformer blocks
        x = self.transformer_blocks(x)

        # final normalization
        x = self.final_norm(x)

        # return probabilities of vocabularies
        logits = self.out_head(x)

        return logits


# note: this is an environment!!!
def generate_text(model, idx, max_new_tokens, context_size):
    """
    Use a provided model as a text generator. Corresponds to greedy sampling from RL
    :param model: the gpt model employed
    :param idx: input encodings
    :param max_new_tokens: the most number of tokens that are to be appended ot the original
    :param context_size: the window of text used within the prediction
    :return:
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size] # obtain the parts of the sequence used for inference,
        with torch.no_grad():
            logs = model(idx_cond)

        logs = logits[:, -1, :] # focuses on the last generated embedding
        probs = torch.softmax(logs, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

if __name__ == "__main__":
    my_config = {
        "vocab_size": 50257,    # the vocabulary size
        "context_length": 1024, # context length for maximum sequence
        "emb_dim": 768,     # the embedding dimension
        "n_heads": 12,      # number of attention heads
        "n_layers": 12,     # number of transformer blocks
        "drop_rate": 0.1,   # rate of dropout
        "qkv_bias": False   # bias associated with Q/K/V matrices
    }

    # inputs
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    my_gpt = BabyGPTModel(my_config)

    torch.manual_seed(123)
    logits = my_gpt(batch)

    # print("Output shape: ", logits.shape)
    # print(logits)

    # proper test
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded tensor shape", encoded_tensor.shape)

    my_gpt.eval()
    out = generate_text(model=my_gpt,
                        idx=encoded_tensor,
                        max_new_tokens=6,
                        context_size=my_config["context_length"])

    print("Output:", out)
    print("Output length:", len(out[0]))