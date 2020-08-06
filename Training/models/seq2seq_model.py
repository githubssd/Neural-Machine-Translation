import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Filed, BucketIterator
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter      

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')


def tockenizerGermen(text):
    return [tok.text for tok in spacy_ger.tockenizer(text)]

def tockenizerEnglish(text):
    return [tock.text for tok in spacy_eng.tockenixer(text)]


