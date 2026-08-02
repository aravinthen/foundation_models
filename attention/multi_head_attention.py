#
# Program name: multi_head_attention.py
# Description: Implementation of a multi-head attention block
#

import torch
import torch.nn as nn

from attention.masked_attention import MaskedAttention
from my_encodings.simple_data import SimpleTokenizer

class NaiveMultiHeadAttention(nn.Module):
    """
    Simply stacks a bunch of attention heads.
    """

    def __init__(self, d_in, d_out, context_len, dropout, num_heads):
        super().__init__()

        self.heads = nn.ModuleList(
            [MaskedAttention(d_in, d_out, context_len, dropout)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads])

class MultiHeadAttention(nn.Module):
    """
    An implementation of multihead attention that uses weight splits
    """

    def __init__(self, d_in, d_out, context_len, dropout, num_heads):
        super().__init__()

        # ensure the context vectors fit together to provide the appropriate multihead output
        assert(d_out % num_heads == 0)

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_len = context_len
        self.dropout = nn.Dropout(dropout)

        # weights
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

        # projection following the multi=head block - uses bias
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer('mask',
                             torch.triu(torch.ones(context_len, context_len, dtype=torch.bool),
                                        diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # split the matrices by adding a num_heads dimension and unrolling the last dimension
        # this will transform the weight matrices into a 4D vector, where the initial 3rd index
        # is partitioned into the correct dimensions for each individual input tensor.
        # this allows the full calculation to be carried out with a single matrix multiplication.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose to set the num_heads before the num_tokes dimensions - allows for correct aligning
        keys = keys.transpose(1,2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # calculate attention scores with masking
        # the matrix multiplication is conducted on the two last dimensions and repeated for individual heads
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vs = (attn_weights @ values).transpose(1,2)
        context_vs = context_vs.contiguous().view(b, num_tokens, self.d_out)

        # linear projection
        context_vs = self.out_proj(context_vs)

        return context_vs


if __name__=="__main__":
    context_length = 6
    batch_size = 2
    embedding_s = 3

    attention_embedding = SimpleTokenizer(embedding_size=embedding_s)
    file_path = "/home/aravinthen/Code/foundation_models/the-verdict.txt"
    attention_embedding.set_vocabulary(file_path)

    single_batch = []
    for i in range(batch_size):
        test = attention_embedding.sample_random_tokens(context_length)
        token_ids = torch.tensor([attention_embedding.vocab[token] for token in test],
                                 dtype=torch.long)

        inputs = attention_embedding.embedding_layer(token_ids)

        single_batch.append(inputs)

    batched_input = torch.stack(single_batch)

    # dims
    batched_input_d= batched_input.shape

    torch.manual_seed(123)

    mh = MultiHeadAttention(embedding_s, 2, context_length,0.2,
                            num_heads=2)

    context_vecs = mh(batched_input)

    print("context_vecs.shape: ", context_vecs.shape)