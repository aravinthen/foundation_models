#
# Program name:     simple_data.py
# Description:      A simple tokenizer and dataset.
#
import re
import torch
from torch.utils.data import Dataset
import urllib.request
from typing import List
import random

class SimpleDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class SimpleTokenizer:
    """
    A very simple tokenizer class, obtained from Raschka's Build a Large
    Language Model textbook.
    """
    def __init__(self, embedding_size = 128):
        self.all_words = None
        self.vocab_size = None
        self.source_text = None
        self.tokenized_source = None

        self.vocab = None
        self.converter = None

        self.embedding_size = embedding_size
        self.embedding_layer = None

    def set_vocabulary(self, file_path, url_path="",):
        """
        Use a URL to obtain a vocabulary text.
        :param url_path: the url of the vocab basis.
        :param file: the file actually being used as the vocabulary.
        """

        if url_path:
            urllib.request.urlretrieve(url_path, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        self.source_text = raw_text
        self.tokenized_source = self.split(self.source_text)

        self.all_words = sorted(set(self.tokenized_source))
        self.all_words.extend(["<|end-of-text|>", "<|unk|>"])

        self.vocab = {token: integer for integer, token in enumerate(self.all_words)}
        self.converter = {integer: token for token, integer in self.vocab.items()}

        self.vocab_size = len(self.vocab)

        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size)


    @staticmethod
    def split(input_text) -> List:
        """
        Splits a string into segments
        :param input_text:
        :return:
        """

        segments = re.split(r'([,.:;?_!"()\']|--|\s)', input_text)
        segments = [item.strip() for item in segments if item.strip()]

        return segments

    def encode(self, input_text, allowed_special=None) -> List:
        """
        Converts a split-list string into tokens
        :param input_text:
        :return:
        """
        segments = self.split(input_text)
        segments = [s if s in self.vocab else "<|unk|>"
                    for s in segments]
        return [self.vocab[s] for s in segments]

    def decode(self, input_tokens):
        """
        Returns the translated input
        :param input_tokens:
        :return:
        """
        return [self.converter[t] for t in input_tokens]

    def sample_random_tokens(self, n) -> list:
        """
        Generates n random tokens
        :param n: number of tokens
        :return: list of tokens
        """
        return random.choices(list(self.vocab), k=n)

if __name__=='__main__':

    vocab = []
    test = SimpleTokenizer()

    # set vocab
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMS-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")

    file_path = '../the-verdict.txt'
    test.set_vocabulary(file_path, url)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    data_loader = SimpleDataset(text, test, 4, stride=4)

    print(test.embedding_layer.weight)