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
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

text_path = 'data/cutted_content/'
tfidf_voc_path = 'data/tfidf_voc.pkl'
label_enc_path = 'data/label_enc.pkl'
tfidf_rank_path = 'data/tfidf_rank.npz'
prop_graph_path = 'data/prop_graph.npz'
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
        # print(doc)
        return doc

    for ddl in [1, 2, 6, 12, 24, 36, 48, 72, 96]:
        docs = []
        for ix, item in enumerate(glob(f'data/cutted_content_{ddl}/*.txt')):
            docs += split_doc(item)
        tfidf = TfidfVectorizer(max_features=max_features).fit_transform(docs)
        print(len(docs), tfidf.shape)
        rank = np.argsort(tfidf.toarray(), axis=1)[:, -top_k:]
        label_enc = LabelEncoder()
        rank = label_enc.fit_transform(rank.reshape(-1)).reshape(-1, top_k)
        joblib.dump(tfidf, tfidf_voc_path)
        joblib.dump(label_enc, label_enc_path)
        print('top K vocsize:', np.max(rank))

        rank_dic = {}
        for ix, item in enumerate(glob(text_path + '*.txt')):
            basename = item.split('/')[-1][:-4]
            rank_dic[basename] = rank[ix * nstage: ix * nstage + nstage]
        np.savez(f'data/tfidf_rank_{ddl}.npz', **rank_dic)


class DSet(Dataset):

    def __init__(self, samples):
        self.data = np.zeros((len(samples), nstage, top_k), dtype=int)
        self.target = np.zeros(len(samples), dtype=np.float32)
        raw_data = np.load(tfidf_rank_path)
        for i, sample in enumerate(samples):
            basename = sample.split('/')[-1][:-5]
            self.data[i] = raw_data[basename]
            if 'rumor' in sample:
                self.target[i] = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, ix):
        return self.data[ix], self.target[ix]


class CDSet(Dataset):

    def __init__(self, samples, step=100, ddl=None):
        self.data_cont = np.zeros((len(samples), nstage, top_k), dtype=int)
        self.data_prop = np.zeros((len(samples), step, 4), dtype=np.float32)
        self.target = np.zeros(len(samples), dtype=np.float32)
        if not ddl:
            raw_data_cont = np.load(tfidf_rank_path)
        else:
            raw_data_cont = np.load(f'data/tfidf_rank_{ddl}.npz')
        raw_data_prop = np.load(prop_graph_path)

        span = 8.1 if not ddl else np.log10(ddl * 3600) + 1
        for i, sample in enumerate(samples):
            basename = sample.split('/')[-1][:-5]
            self.data_cont[i] = raw_data_cont[basename]
            for t, lx, ly, x, y in raw_data_prop[sample]:
                if ly > 5: continue
                if ddl and t > ddl * 3600: continue
                tid = int(np.log10(t + 1) * step / span)
                self.data_prop[i][tid][ly-2] += 1
            if 'rumor' in sample:
                self.target[i] = 1

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, ix):
        return self.data_cont[ix], self.data_prop[ix], self.target[ix]


class CNN(nn.Module):

    def __init__(self, voc_size, emb_size):
        super(CNN, self).__init__()
        self.emb_size = emb_size
        self.emb = nn.Embedding(voc_size, emb_size)
        self.conv1 = nn.Conv2d(1, 8, (7, emb_size), padding=(3, 0))
        self.conv2 = nn.Conv1d(8, 16, 3, padding=1)
        self.fc = nn.Linear(nstage * 16 // 4, 1)

    def forward(self, x):
        x = torch.mean(self.emb(x), dim=2).view(x.size(0), 1, nstage, self.emb_size)
        x = self.conv1(x).view(x.size(0), -1, x.size(2))
        x = F.relu(F.max_pool1d(x, 2))
        x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = F.dropout(x.view(x.size(0), -1), training=self.training)
        x = F.sigmoid(self.fc(x))
        return x


class RNN(nn.Module):

    def __init__(self, voc_size, emb_size, hidden_size=100):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(voc_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        x = torch.mean(self.emb(x), dim=2).view(x.size(0), nstage, -1)
        x, hn = self.rnn(x, h0)
        x = self.fc(x[:, -1, :])
        return F.sigmoid(x)


class NET(nn.Module):

    def __init__(self, voc_size, emb_size, input_size):
        super(NET, self).__init__()
        self.emb = nn.Embedding(voc_size, emb_size)
        self.cconv1 = nn.Conv2d(1, 8, (7, emb_size), padding=(3, 0))
        self.cconv2 = nn.Conv1d(8, 16, 3, padding=1)
        self.cfc = nn.Linear(nstage * 16 // 4, 64)

        self.pconv1 = nn.Conv2d(1, 8, (7, 2), padding=(3, 0))
        self.pconv2 = nn.Conv2d(8, 16, 3, padding=(1, 0))
        self.pfc = nn.Linear(input_size // 4 * 16, 64)

        self.fc = nn.Linear(64 * 2, 1)

    def forward(self, x, y):

        x = torch.mean(self.emb(x), dim=2).view(x.size(0), 1, nstage, -1)
        x = self.cconv1(x).view(x.size(0), -1, x.size(2))
        x = F.relu(F.max_pool1d(x, 2))
        x = F.relu(F.max_pool1d(self.cconv2(x), 2))
        x = F.dropout(x.view(x.size(0), -1), training=self.training)
        x = F.sigmoid(self.cfc(x))

        y = y.view(y.size(0), 1, y.size(1), y.size(2))
        y = F.relu(F.max_pool2d(self.pconv1(y), kernel_size=(2, 1), stride=(2, 1)))
        y = F.relu(F.max_pool2d(self.pconv2(y), kernel_size=(2, 1), stride=(2, 1)))
        y = F.dropout(y.view(y.size(0), -1), training=self.training)
        y = F.relu(self.pfc(y))

        z = self.fc(torch.cat((x, y), 1))
        return F.sigmoid(z)


def train(model, n_epoch=20):

    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.05)

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
            pred = torch.sign(output.cpu() - 0.5).clamp(min=0)
            tr_acc += pred.eq(target.cpu()).sum().item()
        tr_loss /= len(train_loader.dataset)
        tr_acc /= len(train_loader.dataset)
        record['tr_loss'].append(tr_loss)
        record['tr_acc'].append(tr_acc)
        print(f'tr_loss {tr_loss:.6f} | tr_acc {tr_acc*100:.2f}%')

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        record['predict'] = []
        with torch.no_grad():
            for data, target in test_loader:
                target = target.view(target.size(0), 1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                pred = torch.sign(output.cpu() - 0.5).clamp(min=0)
                val_acc += pred.eq(target.cpu()).sum().item()
                record['predict'].append(output.cpu().numpy())
        val_loss /= len(test_loader.dataset)
        val_acc /= len(test_loader.dataset)
        record['val_loss'].append(val_loss)
        record['val_acc'].append(val_acc)
        print(f'val_loss {val_loss:.6f} | val_acc {val_acc*100:.2f}%')
        if record['val_acc'][-1] > acc_max:
            acc_max = record['val_acc'][-1]
            record['final'] = np.vstack(record['predict']).reshape(-1)
    return record


def xtrain(model, n_epoch=5):

    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.05)

    print(f'training {model.__class__.__name__} ...')
    acc_max = 0.0
    record = {x: list() for x in ['tr_loss', 'tr_acc', 'val_loss', 'val_acc', 'predict']}
    for epoch in range(n_epoch):

        model.train()
        tr_loss, tr_acc = 0.0, 0.0
        for datac, datap, target in train_loader:
            target = target.view(target.size(0), 1)
            datac, datap, target = datac.to(device), datap.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(datac, datap)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * datac.size(0)
            pred = torch.sign(output.cpu() - 0.5).clamp(min=0)
            tr_acc += pred.eq(target.cpu()).sum().item()
        tr_loss /= len(train_loader.dataset)
        tr_acc /= len(train_loader.dataset)
        record['tr_loss'].append(tr_loss)
        record['tr_acc'].append(tr_acc)
        # print(f'Epoch {(epoch + 1):02d} | tr_loss {tr_loss:.6f} | tr_acc {tr_acc*100:.2f}%')

    model.eval()
    val_loss, val_acc = 0.0, 0.0
    record['predict'] = []
    with torch.no_grad():
        for datac, datap, target in test_loader:
            target = target.view(target.size(0), 1)
            datac, datap, target = datac.to(device), datap.to(device), target.to(device)
            output = model(datac, datap)
            loss = criterion(output, target)
            val_loss += loss.item() * datac.size(0)
            pred = torch.sign(output.cpu() - 0.5).clamp(min=0)
            val_acc += pred.eq(target.cpu()).sum().item()
            record['predict'].append(output.cpu().numpy())
    val_loss /= len(test_loader.dataset)
    val_acc /= len(test_loader.dataset)
    record['val_loss'].append(val_loss)
    record['val_acc'].append(val_acc)
    record['predict'] = np.vstack(record['predict']).reshape(-1)
    # print(f'val_loss {val_loss:.6f} | val_acc {val_acc*100:.2f}%')
    return record


def census(output, target):

    from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

    tpr, fpr, _ = roc_curve(target, output)
    auc = roc_auc_score(target, output)
    output[output < 0.5] = 0
    output[output > 0.4] = 1
    rp = precision_score(target, output, pos_label=1)
    rr = recall_score(target, output, pos_label=1)
    rf = f1_score(target, output, pos_label=1)
    np = precision_score(target, output, pos_label=0)
    nr = recall_score(target, output, pos_label=0)
    nf = f1_score(target, output, pos_label=0)
    acc = accuracy_score(target, output)
    print(f'acc: {acc:.3f}\nrp: {rp:.3f}\nrr: {rr:.3f}\nrf: {rf:.3f}')
    print(f'np: {np:.3f}\nnr: {nr:.3f}\nnf: {nf:.3f}')
    return tpr, fpr, auc


def get_weights(total):

    for i in range(total):
        model = NET(max_features, 100, 100)
        rec = xtrain(model)
        pweight = model.pconv1.weight.detach().cpu().numpy().reshape(8, 7, 2)
        cweight = model.cconv1.weight.detach().cpu().numpy().reshape(8, 7, 100)
        cweight = np.sum(cweight, axis=2).reshape(8, 7)
        np.savez(f'heatmaps/{i}.npz', pweight=pweight, cweight=cweight)


def draw_checkboard(total):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    for i in range(total):

        data = np.load(f'heatmaps/{i}.npz')
        pweight, cweight = data['pweight'], data['cweight']

        plt.figure()

        f, ax = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(pd.DataFrame(pweight[:, :, 0]), ax=ax[0])
        sns.heatmap(pd.DataFrame(pweight[:, :, 1]), ax=ax[1])
        plt.savefig(f'heatmapp/{i}.png', dpi=200)

        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.heatmap(pd.DataFrame(cweight), ax=ax)
        plt.savefig(f'heatmapc/{i}.png', dpi=200)

        print('done for', i)


if __name__ == '__main__':

    # print('loading data ...')
    # samples = glob('rumor/*.json') + glob('truth/*.json')
    # train_data, test_data = train_test_split(samples, test_size=0.2, random_state=42)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # train_loader = DataLoader(CDSet(train_data, 100), batch_size=128, shuffle=True, **kwargs)
    # test_loader = DataLoader(CDSet(test_data, 100), batch_size=128, **kwargs)

    draw_checkboard(500)



