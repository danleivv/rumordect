#!/usr/bin/env python3.6

import json
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

use_gpu = torch.cuda.is_available()


class DSet(Dataset):

    def __init__(self, samples, step=100):
        self.data = np.zeros((len(samples), step), dtype=np.float32)
        self.target = np.zeros(len(samples), dtype=np.float32)
        raw_data = np.load('data/prop_span.npz')
        for i, sample in enumerate(samples):
            span = raw_data[sample]
            volumn = np.log10(span + 1) / 8.1 * step
            for item in volumn.astype(int):
                self.data[i][item] += 1
            if 'rumor' in sample:
                self.target[i] = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class CNN(nn.Module):

    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(input_size // 4 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = F.dropout(x.view(x.size(0), -1), training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=64, bidirectional=True):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_directions = 2 if bidirectional else 1
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.n_directions, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.input_size)
        h0 = self._init_hidden_state(x.size(0))
        x, hn = self.rnn(x, h0)
        x = self.fc(x[:, -1, :])
        return F.sigmoid(x)

    def _init_hidden_state(self, batch_size):
        h0 = torch.zeros(self.n_directions, batch_size, self.hidden_size)
        if use_gpu:
            h0 = h0.cuda()
        return Variable(h0)


class CombinedNet(nn.Module):

    def __init__(self, cnn_input_size, rnn_input_size, hidden_size=64):
        super(CombinedNet, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_input_size = rnn_input_size
        self.rnn = nn.GRU(rnn_input_size, hidden_size, batch_first=True, bidirectional=True, dropout=0.5)
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1)
        self.fc_dim = cnn_input_size // 4 * 16 + hidden_size * 2
        self.fc = nn.Linear(self.fc_dim, 1)

    def forward(self, x):
        rx = x.view(x.size(0), -1, self.rnn_input_size)
        cx = x.view(x.size(0), 1, -1)
        h0 = self._init_hidden_state(rx.size(0))
        rx, hn = self.rnn(rx, h0)
        cx = F.relu(F.max_pool1d(self.conv1(cx), 2))
        cx = F.relu(F.max_pool1d(self.conv2(cx), 2))
        rcx = torch.cat((rx[:, -1, :].view(x.size(0), -1), cx.view(x.size(0), -1)), dim=1)
        out = self.fc(F.dropout(rcx, training=self.training))
        return F.sigmoid(out)

    def _init_hidden_state(self, batch_size):
        h0 = torch.zeros(2, batch_size, self.hidden_size)
        if use_gpu:
            h0 = h0.cuda()
        return Variable(h0)


def train(model, n_epoch=10):

    if use_gpu:
        model.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())

    print(f'training {model.__class__.__name__} ...')
    record = {x: list() for x in ['tr_loss', 'tr_acc', 'val_loss', 'val_acc']}
    for epoch in range(n_epoch):
        print(f'Epoch {(epoch + 1):02d}')
        model.train()
        tr_loss, tr_acc = 0.0, 0.0
        for data, target in train_loader:
            target = target.view(target.size(0), 1)
            optimizer.zero_grad()
            if use_gpu:
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
            if use_gpu:
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
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    train_loader = DataLoader(DSet(train_data), batch_size=128, shuffle=True, **kwargs)
    test_loader = DataLoader(DSet(test_data), batch_size=128, **kwargs)

    rec = train(CombinedNet(100, 10), 10)
