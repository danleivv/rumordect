#!/usr/bin/env python3

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob


use_gpu = torch.cuda.is_available()


class DSet(Dataset):

    def __init__(self, samples, step=100):
        self.data = np.zeros((len(samples), step))
        self.target = np.zeros(len(samples), dtype=np.float32)
        raw_data = json.load(open('data/prop_span.json'))
        for i, sample in enumerate(samples):
            span = raw_data[sample]
            volumn = map(lambda x: int(x * step / 8.1), np.log10(span))
            for item in volumn:
                self.data[i][item] += 1
            if 'rumor' in sample:
                self.target[i] = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), self.target[idx]


class CNN(torch.nn.Module):

    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1)
        self.fc_in = input_size // 4 * 16
        self.fc1 = nn.Linear(self.fc_in, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = F.dropout(x.view(-1, self.fc_in), training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)


class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size=64, bidirectional=True):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_directions = 2 if bidirectional else 1
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * self.n_directions, 1)

    def forward(self, x):
        h0 = self._init_hidden_state(x.size(0))
        x, hn = self.rnn(x, h0)
        x = self.fc(x[:, -1, :])
        return F.sigmoid(x)

    def _init_hidden_state(self, batch_size):
        h0 = torch.zeros(self.n_directions, batch_size, self.hidden_size)
        if use_gpu:
            h0 = h0.cuda()
        return Variable(h0)


def train_cnn(step=100, n_epoch=10):

    print('building model ...')
    model = CNN(step)
    if use_gpu:
        model.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())

    for epoch in range(n_epoch):
        print('Epoch %03d:' % (epoch + 1))
        train_loss, test_loss, tr_acc, val_acc = 0.0, 0.0, 0.0, 0.0
        model.train()
        for data, target in train_loader:
            data = data.view(data.size(0), 1, data.size(1))
            target = target.view(target.size(0), 1)
            optimizer.zero_grad()
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0] * data.size(0)
            pred = torch.sign(output.data - 0.5).clamp_(min=0)
            tr_acc += pred.eq(target.data).cpu().sum()
        train_loss /= len(train_loader.dataset)
        tr_acc = tr_acc / len(train_loader.dataset) * 100
        print('tr_loss %.6f | tr_acc %.2f%%' % (train_loss, tr_acc))

        model.eval()
        for data, target in test_loader:
            data = data.view(data.size(0), 1, data.size(1))
            target = target.view(target.size(0), 1)
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.data[0] * data.size(0)
            pred = torch.sign(output.data - 0.5).clamp_(min=0)
            val_acc += pred.eq(target.data).cpu().sum()
        test_loss /= len(test_loader.dataset)
        val_acc = val_acc / len(test_loader.dataset) * 100
        print('val_loss %.6f | val_acc %.2f%%' % (test_loss, val_acc))


def train_rnn(step_size=10, n_epoch=10):

    print('building model ...')
    model = RNN(step_size)
    if use_gpu:
        model.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())

    for epoch in range(n_epoch):
        print('Epoch %03d:' % (epoch + 1))
        train_loss, test_loss, tr_acc, val_acc = 0.0, 0.0, 0.0, 0.0
        model.train()
        for data, target in train_loader:
            data = data.view(data.size(0), -1, step_size)
            target = target.view(target.size(0), 1)
            optimizer.zero_grad()
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0] * data.size(0)
            pred = torch.sign(output.data - 0.5).clamp_(min=0)
            tr_acc += pred.eq(target.data).cpu().sum()
        train_loss /= len(train_loader.dataset)
        tr_acc = tr_acc / len(train_loader.dataset) * 100
        print('tr_loss %.6f | tr_acc %.2f%%' % (train_loss, tr_acc))

        model.eval()
        for data, target in test_loader:
            data = data.view(data.size(0), -1, step_size)
            target = target.view(target.size(0), 1)
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.data[0] * data.size(0)
            pred = torch.sign(output.data - 0.5).clamp_(min=0)
            val_acc += pred.eq(target.data).cpu().sum()
        test_loss /= len(test_loader.dataset)
        val_acc = val_acc / len(test_loader.dataset) * 100
        print('val_loss %.6f | val_acc %.2f%%' % (test_loss, val_acc))


if __name__ == '__main__':

    print('loading data ...')
    samples = glob('rumor/*.json') + glob('truth/*.json')
    train_data, test_data = train_test_split(samples, test_size=0.2, random_state=42)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    train_loader = DataLoader(DSet(train_data), batch_size=128, **kwargs)
    test_loader = DataLoader(DSet(test_data), batch_size=128, **kwargs)

    train_rnn(10, 10)