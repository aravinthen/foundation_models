#
# Program name: scaled_dot_product_attn.py
# Description: A proper implementation of scaled dot-product attention
#

from my_encodings.simple_data import SimpleTokenizer
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Implementation of a simple attention mechanism.
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key =  nn.Linear(d_in, d_out, bias=False)
        self.W_value =  nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        """
        Generic forward pass with attention weights
        :param x: A tensor of shapes
        :return:
        """
        ks = self.W_key(x)
        qs =  self.W_query(x)
        vals = self.W_value(x)

        attn_scores = qs @ ks.T
        attn_weights = torch.softmax(attn_scores/ks.shape[-1]**0.5, dim=-1)

        context_vecs = attn_weights @ vals

        return context_vecs

if __name__ == "__main__":

    attention_embedding = SimpleTokenizer(embedding_size=4)
    file_path = "/home/aravinthen/Code/foundation_models/the-verdict.txt"
    attention_embedding.set_vocabulary(file_path)

    test = attention_embedding.sample_random_tokens(5)
    token_ids = torch.tensor([attention_embedding.vocab[token] for token in test],
                             dtype=torch.long)

    inputs = attention_embedding.embedding_layer(token_ids)

    # dims
    d_in = inputs.shape[1]
    d_out = 2

    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)

    queries = inputs @ W_query
    keys = inputs @ W_value
    values = inputs @ W_key

    attention_scores = queries @ keys.T

    d_k = keys.shape[-1]
    attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)

    context_vectors = attention_weights @ values

    print("Tokens:")
    print(test)

    print("\nInput embeddings:")
    print(inputs)

    print("\nQueries:")
    print(queries)

    print("\nKeys:")
    print(keys)

    print("\nValues:")
    print(values)

    print("\nAttention scores:")
    print(attention_scores)

    print("\nAttention weights:")
    print(attention_weights)

    print("\nAttention-weight row sums:")
    print(attention_weights.sum(dim=-1))

    print("\nContext vectors:")
    print(context_vectors)
