#!/usr/bin/env python3.6

from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2vec, LabeledSentence

use_gpu = torch.cuda.is_available()


def doc2vec(stage=20, dim=100, vocab_size=10000):

    def split_doc(item):
        basename = item.split('/')[-1][:-3]
        doc = open(item).readlines()
        splitted = []
        span = (len(doc)) // stage
        rest = (len(doc)) - span * n_stage
        start = 0
        for i in range(rest):
            sentence = ''.join(doc[start:start+span+1]).replace('\n', ' ')
            labeled = LabeledSentence(sentence, [basename + str(i)])
            splitted.append(labeled)
            start += span + 1
        if span:
            for i in range(stage - rest):
                sentence = ''.join(doc[start:start+span]).replace('\n', '')
                labeled = LabeledSentence(sentence, [basename + str(i + rest)])
                splitted.append(labeled)
                start += span
        return splitted

    labeled_doc = []
    for item in glob('data/tokenized_content/*.txt'):
        labeled_doc.append(split_doc(item))
    model = Doc2Vec(labeled_doc, size=100, window=8, min_count=5, workers=4)


class RNN(nn.Module):

    def __init__(self, vocab_size, input_size, hidden_size=128, bidirectional=True):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_directions = 2 if bidirectional else 1
        self.emb = nn.Embedding(vocab_size + 1, input_size)
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, 1)

    def forward(self, x):
        x = self.emb(x.clamp(max=self.vocab_size))
        x, hn = self.rnn(x, self._init_hidden_state(x.size(0)))
        x = self.fc(x[:, -1, :])
        return F.sigmoid(x)

    def _init_hidden_state(self, batch_size)
        h0 = torch.zeros(self.n_directions, batch_size, self.hidden_size)
        if use_gpu:
            h0 = h0.cuda()
        return Variable(h0)


if __name__ == '__main__':

    doc2vec()