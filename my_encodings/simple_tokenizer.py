#
# Program name:     simple_tokenizer.py
# Description:      A library of simple tokenizers.
#
import re
import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request
from typing import List
import random

class SimpleDataset(Dataset):
    """
    A simple dataset loader class -  taken from Raschka directly.
    """

    def __init__(self, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        self.tokenizer = tokenizer

        token_ids = tokenizer.encode(tokenizer.source_text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1: i+max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def create_data_loader(self,
                           batch_size=4,
                           shuffle=True,
                           drop_last=True,
                           num_workers=0):

        tokenizer = self.tokenizer
        dataset = self

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader

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
            raw_test = f.read()

        self.source_text = raw_test
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

    def encode(self, input_text) -> List:
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
    data_loader = SimpleDataset(test, 10, 4).create_data_loader()

    print(test.embedding_layer.weight)