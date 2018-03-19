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
from sklearn.ensemble import RandomForestClassifier


use_cuda = torch.cuda.is_available()
STAGE = 20
VEC_SIZE = 100
D2V_EPOCH = 750


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


def doc2vec(epochs=1000, *, stage=STAGE, vector_size=VEC_SIZE, vocab_size=10000):

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


class DSet(Dataset):

    def __init__(self, samples, epochs=D2V_EPOCH, stage=STAGE, vector_size=VEC_SIZE):
        self.data = np.zeros((len(samples), stage, vector_size), dtype=np.float32)
        self.target = np.zeros(len(samples), dtype=np.float32)
        model = Doc2Vec.load(f'data/doc2vec_{epochs}.model')
        for i, sample in enumerate(samples):
            basename = sample.split('/')[-1][:-4]
            for j in range(stage):
                label = basename + str(j)
                if label in model.docvecs:
                    self.data[i, j] = model.docvecs[label]
            if 'rumor' in sample:
                self.target[i] = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def stacking(epochs=750, stage=20):

    model = Doc2Vec.load(f'data/doc2vec_{epochs}.model')
    X, y = [], []
    samples = glob('rumor/*.json') + glob('truth/*.json')
    for item in samples:
        basename = item.split('/')[-1][:-4]
        for i in range(stage):
            label = basename + str(i)
            if label in model.docvecs:
                X.append(model.docvecs[label])
                y.append(1 if 'rumor' in item else 0)
    X = np.vstack(X)
    clf = LogisticRegression()
    clf.fit(X, y)
    X, y = [], []
    data_loader = DataLoader(DSet(samples, epochs))
    for data, target in data_loader:
        data = data.numpy().reshape(stage, -1)
        X.append(clf.predict_proba(data)[:, 0])
        y.append(target.numpy())
    y = np.hstack(y)
    clf = LogisticRegression()
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
    clf.fit(X_tr, y_tr)
    print(clf.score(X_tr, y_tr))
    print(clf.score(X_val, y_valst))


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
        if use_cuda:
            h0 = h0.cuda()
        return Variable(h0)


class CNN(nn.Module):

    def __init__(self, input_h, input_w):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, input_w), padding=(2, 0))
        self.conv2 = nn.Conv1d(4, 6, 7, padding=3)
        self.fc1 = nn.Linear(input_h // 4 * 6, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.conv1(x).view(x.size(0), -1, x.size(2))
        x = F.relu(F.max_pool1d(x, 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = F.dropout(x.view(x.size(0), -1), training=self.training)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)


def train(model, epochs=100):

    if use_cuda:
        model.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())

    print(f'training {model.__class__.__name__} ...')
    record = {x: list() for x in ['tr_loss', 'tr_acc', 'val_loss', 'val_acc']}
    for epoch in range(epochs):
        print(f'Epoch {(epoch + 1):02d}')
        model.train()
        tr_loss, tr_acc = 0.0, 0.0
        for data, target in train_loader:
            target = target.view(target.size(0), 1)
            optimizer.zero_grad()
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            tr_loss += loss.data[0] * data.size(0)
            pred = torch.sign(output.data - 0.5).clamp(min=0)
            tr_acc += pred.eq(target.data).cpu().sum()
        tr_loss /= len(train_loader.dataset)
        tr_acc = tr_acc / len(train_loader.dataset)
        record['tr_loss'].append(tr_loss)
        record['tr_acc'].append(tr_acc)
        print(f'tr_loss {tr_loss:.6f} | tr_acc {tr_acc*100:.2f}%')

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        for data, target in test_loader:
            target = target.view(target.size(0), 1)
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.data[0] * data.size(0)
            pred = torch.sign(output.data - 0.5).clamp(min=0)
            val_acc += pred.eq(target.data).cpu().sum()
        val_loss /= len(test_loader.dataset)
        val_acc = val_acc / len(test_loader.dataset)
        record['val_loss'].append(val_loss)
        record['val_acc'].append(val_acc)
        print(f'val_loss {val_loss:.6f} | val_acc {val_acc*100:.2f}%')
    return record


if __name__ == '__main__':

    print('loading data ...')
    samples = glob('rumor/*.json') + glob('truth/*.json')
    train_data, test_data = train_test_split(samples, test_size=0.2, random_state=42)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(DSet(train_data), batch_size=128, shuffle=True, **kwargs)
    test_loader = DataLoader(DSet(test_data), batch_size=128, **kwargs)
    rec = train(CNN(STAGE, VEC_SIZE))