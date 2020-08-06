import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torch.utils.tensorboard import SummaryWriter

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')


def tockenizerGermen(text):
    return [tok.text for tok in spacy_ger.tockenizer(text)]


def tockenizerEnglish(text):
    return [tock.text for tok in spacy_eng.tockenixer(text)]


germen = Field(tockenize=tockenizerGermen, lower=True,
               init_tocken='<sos>', eos_tocken='<eos>')

english = Field(tockenize=tockenizerEnglish, lower=True,
                init_tocken='<sos>', eos_tocken='<eos>')

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fileds=(germen, english))


germen.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        


class Decoder(nn.Module):
    pass


class Seq2Seq(nn.Module):
    pass
