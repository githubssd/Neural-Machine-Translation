import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')


def tockenizerGermen(text):
    return [tok.text for tok in spacy_ger.tockenizer(text)]


def tockenizerEnglish(text):
    return [tock.text for tok in spacy_eng.tockenixer(text)]


german = Field(tockenize=tockenizerGermen, lower=True,
               init_tocken='<sos>', eos_tocken='<eos>')

english = Field(tockenize=tockenizerEnglish, lower=True,
                init_tocken='<sos>', eos_tocken='<eos>')

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fileds=(german, english))


german.build_vocab(train_data, max_size=10000, min_freq=2)
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
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):

        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size,
                              target_vocab_size).to(device)
        hidden, cell = self.encoder(source)

        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output

            best_guess = output.argmax(1)
            x = target[t] if random.random(
            ) < teacher_force_ratio else best_guess

        return outputs


# Training
num_epochs = 50
lr = 0.001
batch_size = 64

load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)

output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embwdding_size = 300
hidden_size = 1024
num_layers = 4

enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embwdding_size,
                      hidden_size, output_size, num_layers, dec_dropout).to(device)


model = Seq2Seq(encoder_net, decoder_net).to(device)

pad_idx = english.vocab.stoi('<pad>')
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}'])

    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer}
