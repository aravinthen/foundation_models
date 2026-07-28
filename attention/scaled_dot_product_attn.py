#
# Program name: scaled_dot_product_attn.py
# Description: A proper implementation of scaled dot-product attention
#

from my_encodings.simple_tokenizer import SimpleTokenizer

import torch
import torch.nn as nn

attention_embedding = SimpleTokenizer(embedding_size=4)
file_path = "/home/aravinthen/Code/foundation_models/the-verdict.txt"
attention_embedding.set_vocabulary(file_path)

test = attention_embedding.sample_random_tokens(5)
token_ids = torch.tensor([attention_embedding.vocab[token] for token in test],
                         dtype=torch.long)

inputs = attention_embedding.embedding_layer(token_ids)