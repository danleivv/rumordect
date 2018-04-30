#!/usr/bin/env python3.6

from glob import glob
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib


assert torch.__version__ >= '0.4.0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_path = 'data/cutted_content'
tfidf_voc_path = 'data/tfidf_voc.pkl'
label_enc_path = 'data/label_enc.pkl'
nstage = 20
max_features = 10000
top_k = 10


def make_tfidf():

    def split_doc(filename):
        raw = open(filename).readlines()
        doc = []
        batched = 0
        for i in range(20):
            size = ceil((len(raw) - batched) / (20 - i))
            doc.append(''.join(raw[batched:batched+size]))
            batched += size
        return doc

    docs = []
    for ix, item in enumerate(glob(text_path + '*.txt')):
        docs += split_doc(item)
    tfidf = TfidfVectorizer(max_features=max_features).fit_transform(docs)
    rank = np.argmax(tfidf.toarray(), 1)[:, -top_k:]
    label_enc = LabelEncoder()
    rank = label_enc.fit_transform(rank.reshape(-1)).reshape(-1, top_k)
    joblib.dump(tfidf, tfidf_voc_path)
    joblib.dump(label_enc, label_enc_path)

    rank_dic = {}
    for ix, item in enumerate(glob(text_path + '*.txt')):
        basename = item.split('/')[-1][:-4]
        rank_dic[basename] = rank[ix * nstage: ix * nstage + nstage]
    np.savez('data/tfidf_rank.npz', **rank_dic)


class DSet(Dataset):

    def __init__(self, samples, step=100):
        self.data = np.zeros((len(samples), nstage, top_k), dtype=int)
        self.target = np.zeros(len(samples), dtype=np.float32)
        raw_data = np.load('data/tfidf_rank.npz')
        for i, sample in enumerate(samples):
            basename = sample.split('/')[-1][:-5]
            self.data[i] = raw_data[basename]
            if 'rumor' in sample:
                self.target[i] = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class CNN(nn.Module):

    def __init__(self, nstage, voc_size, emb_size):
        super(CNN, self).__init__()
        self.nstage = nstage
        self.emb_size = emb_size
        self.emb = nn.Embedding(voc_size, emb_size)
        self.conv1 = nn.Conv2d(1, 8, (7, emb_size), padding=(3, 0))
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1)
        self.fc = nn.Linear(nstage * 16 // 4, 1)

    def forward(self, x):
        x = torch.mean(self.emb(x), dim=2).view(x.size(0), 1, nstage, self.emb_size)
        x = self.conv1(x).view(x.size(0), -1, x.size(1))
        x = F.relu(F.max_pool1d(x, 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = F.dropout(x.view(x.size(0), -1), training=self.trianing)
        x = F.sigmoid(self.fc(x))
        return x


def train(model, epoch=20):

    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())

    print(f'training {model.__class__.__name__} ...')
    acc_max = 0.0
    record = {x: list() for x in ['tr_loss', 'tr_acc', 'val_loss', 'val_acc', 'predict']}
    for epoch in range(n_epoch):
        print(f'Epoch {(epoch + 1):02d}')
        model.train()
        tr_loss, tr_acc = 0.0, 0.0
        for data, target in train_loader:
            target = target.view(target.size(0), 1)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * data.size(0)
            pred = torch.sign(output.data - 0.5).clamp(min=0)
            tr_acc += pred.eq(target.data).cpu().sum()
        tr_loss /= len(train_loader.dataset)
        tr_acc = tr_acc / len(train_loader.dataset)
        record['tr_loss'].append(tr_loss)
        record['tr_acc'].append(tr_acc)
        print(f'tr_loss {tr_loss:.6f} | tr_acc {tr_acc*100:.2f}%')

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        record['predict'] = []
        with torch.no_grad:
            for data, target in test_loader:
                target = target.view(target.size(0), 1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                pred = torch.sign(output.data - 0.5).clamp(min=0)
                val_acc += pred.eq(target.data).cpu().sum()
                record['predict'].append(output.data.numpy())
        val_loss /= len(test_loader.dataset)
        val_acc = val_acc / len(test_loader.dataset)
        record['val_loss'].append(val_loss)
        record['val_acc'].append(val_acc)
        print(f'val_loss {val_loss:.6f} | val_acc {val_acc*100:.2f}%')
        if record['val_acc'][-1] > acc_max:
            acc_max = record['val_acc'][-1]
            record['final'] = np.vstack(record['predict']).reshape(-1)
    return record


if __name__ == '__main__':

    make_tfidf()
