#
# Program name: simple_attn.py
# Description: A slightly more streamlined version of simple attention in Raschka's Build a Large Language Model book
#

from my_encodings.simple_tokenizer import SimpleTokenizer

import torch
import torch.nn as nn

attention_embedding = SimpleTokenizer(embedding_size=4)
file_path = "/home/aravinthen/Code/foundation_models/the-verdict.txt"
attention_embedding.set_vocabulary(file_path)

test = attention_embedding.sample_random_tokens(5)
token_ids = torch.tensor([attention_embedding.vocab[token]
                          for token in test],
                         dtype=torch.long)

inputs = attention_embedding.embedding_layer(token_ids)

attention_scores = inputs @ inputs.T
softmax = nn.Softmax(dim=-1)
attention_weights = softmax(attention_scores)

context_vectors = attention_weights @ inputs

print("Tokens:", test)
print("Token IDs:", token_ids)
print("Inputs shape:", inputs.shape)
print("Attention scores:", attention_scores)
print("Attention weights:", attention_weights)
print("Weight sum:", attention_weights.sum(dim=-1))
print("Context vector:", context_vectors)
print("Context vector shape:", context_vectors.shape)