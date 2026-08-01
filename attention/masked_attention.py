#
# Program name: scaled_dot_product_attn.py
# Description: A proper implementation of scaled dot-product attention
#

from my_encodings.simple_tokenizer import SimpleTokenizer
import torch
import torch.nn as nn

class MaskedAttention(nn.Module):
    """
    Implementation of a simple attention mechanism.
    """
    def __init__(self, d_in, d_out, context_length, dropout):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key =  nn.Linear(d_in, d_out, bias=False)
        self.W_value =  nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        # a new technique
        # Register the causal mask so it moves with the module between devices
        # and is included in the module's state.
        self.register_buffer('mask',
                             torch.triu(torch.ones(context_length, context_length, dtype=torch.bool),
                                        diagonal=1))

    def forward(self, x):
        """
        Generic forward pass with attention weights
        :param x: A tensor of shapes
        :return:
        """
        b, num_tokens, d_in = x.shape
        ks = self.W_key(x)
        qs =  self.W_query(x)
        vals = self.W_value(x)

        attn_scores = qs @ ks.transpose(1, 2)

        # carry out the masking
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores/ks.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = attn_weights @ vals

        return context_vecs

if __name__ == "__main__":

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

    ca = MaskedAttention(embedding_s, 2, context_length, 0.2)
    context_vecs = ca(batched_input)

    print("context_vecs.shape: ", context_vecs.shape)
