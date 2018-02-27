#!/usr/bin/env python3.6

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob


class RNN(nn.Module):

    def __init__(self, vocab_size, input_size, hidden_size=128, bidirectional=True):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_directions = 2 if bidirectional else 1
        self.emb = nn.Embedding(vocab_size, input_size)
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, 1)

    def forward(self, x):
        x = self.emb(x)
        x, hn = self.rnn(x, self._init_hidden_state(x.size(0)))
        x = self.fc(x[:, -1, :])
        return F.sigmoid(x)

    def _init_hidden_state(self, batch_size)
        h0 = torch.zeros(self.n_directions, batch_size, self.hidden_size)
        if use_gpu:
            h0 = h0.cuda()
        return Variable(h0)