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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.linear_model import LogisticRegression

use_gpu = torch.cuda.is_available()


class TestLogger(CallbackAny2Vec):

    def __init__(self, stage):
        self.epochs = 0
        self.stage = stage

    def on_epoch_end(self, model):
        self.epochs += 1
        # print(self.epochs)
        if self.epochs % 50 == 0:
            X, y = [], []
            for item in (glob('rumor/*.json') + glob('truth/*.json')):
                basename = item.split('/')[-1][:-4]
                for i in range(self.stage):
                    label = basename + str(i)
                    if label in model.docvecs:
                        X.append(model.docvecs[label])
                        y.append(1 if 'rumor' in item else 0)
            X = np.vstack(X)
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
            clf = LogisticRegression()
            clf.fit(X_tr, y_tr)
            print(f'on epoch {self.epochs}:')
            print(clf.score(X_tr, y_tr))
            print(clf.score(X_val, y_val))
            model.save(f'data/doc2vec_{self.epochs}.model')


def doc2vec(epochs=100, *, stage=20, vector_size=100, vocab_size=10000):

    def split_doc(item):
        basename = item.split('/')[-1][:-3]
        doc = open(item).readlines()
        splitted = []
        span = (len(doc)) // stage
        rest = (len(doc)) - span * stage
        start = 0
        for i in range(rest):
            sentence = ''.join(doc[start:start+span+1]).replace('\n', ' ')
            labeled = TaggedDocument(sentence, [basename + str(i)])
            splitted.append(labeled)
            start += span + 1
        if span:
            for i in range(stage - rest):
                sentence = ''.join(doc[start:start+span]).replace('\n', ' ')
                labeled = TaggedDocument(sentence, [basename + str(i + rest)])
                splitted.append(labeled)
                start += span
        return splitted

    labeled_doc = []
    for item in glob('data/tokenized_content/*.txt'):
        labeled_doc += split_doc(item)
    model = Doc2Vec(vector_size=vector_size, window=8, min_count=5, workers=6)
    model.build_vocab(labeled_doc)
    test = TestLogger(stage)
    model.train(labeled_doc, epochs=epochs, start_alpha=0.025, end_alpha=0.005, total_examples=model.corpus_count, callbacks=[test])
    model.save(f'data/doc2vec_{epochs}.model')


def get_docvec(epochs):

    model = Doc2Vec.load(f'data/doc2vec_{epochs}.model')



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

    def _init_hidden_state(self, batch_size):
        h0 = torch.zeros(self.n_directions, batch_size, self.hidden_size)
        if use_gpu:
            h0 = h0.cuda()
        return Variable(h0)


if __name__ == '__main__':

    doc2vec(300)