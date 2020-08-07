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
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):

        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedding)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size, p):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedding)

        predictions = self.fc(output)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    pass
