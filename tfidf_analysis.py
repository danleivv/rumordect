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


assert torch.__version__ >= '0.4.0'
use_cuda = torch.cuda.is_available()

text_path = None
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
    rank_dic = {}
    for ix, item in enumerate(glob(text_path + '*.txt')):
        basename = item.split('/')[-1][:-3]
        for i in range(20):
            rank_dic[basename + str(i)] = rank[ix * 20 + i]
    np.savez('data/tfidf_rank.npz', **rank_dic)


make_tfidf()
