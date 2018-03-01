#!/usr/bin/env python3.6
import os
import json
from glob import glob

def classify_json():

    for line in open('data/Weibo.txt').readlines():
        idx, label = line.split()[:2]
        fname = idx[4:] + '.json'
        label = label[-1]
        if label == '0':
            os.system('mv Weibo/%s truth/' % fname)
        else:
            os.system('mv Weibo/%s rumor/' % fname)


def get_prop_info():

    import numpy as np

    data = dict()
    for item in (glob('rumor/*.json') + glob('truth/*.json')):
        raw = json.load(open(item))
        span = np.array(list(map(lambda x: x['t'], raw)))
        data[item] = span - np.min(span)
    np.savez('data/prop_span.npz', **data)


def get_content():

    import thulac

    thul = thulac.thulac(seg_only=True)
    os.mkdir('data/original_content')
    os.mkdir('data/cutted_content')
    for item in (glob('rumor/*.json') + glob('truth/*.json')):
        fid = os.path.basename(item)[:-5]
        original = f'data/original_content/{fid}.txt'
        cutted = f'data/cutted_content/{fid}.txt'
        raw = json.load(open(item))
        data = list(map(lambda x: x['text'] + '\n', raw))
        with open(original, 'w') as f:
            f.writelines(data)
        thul.cut_f(original, cutted)


def tokenize():

    from collections import Counter

    counter = Counter()
    vocab = {}
    if not os.path.exists('data/tokenized_content'):
        os.mkdir('data/tokenized_content')
    for item in glob('data/cutted_content/*.txt'):
        for line in open(item).readlines():
            counter.update(line.split())
    for idx, (word, cnt) in enumerate(counter.most_common()):
        vocab[word] = idx
    for item in glob('data/cutted_content/*.txt'):
        tokens = ''
        for line in open(item).readlines():
            line = map(lambda x: str(vocab[x]), line.split())
            tokens += ' '.join(line) + '\n'
        with open(item.replace('cutted', 'tokenized'), 'w') as f:
            f.write(tokens)
    json.dump(vocab, open('data/word2idx.json', 'w'), indent=4)


def replace_lowfrq(k=10000):

    os.mkdir(f'data/tokenized_{k}')
    for item in glob('data/tokenized_content/*.txt'):
        tokens = ''
        for line in open(item).readlines():
            line = map(lambda x: str(min(k, int(x))), line.split())
            tokens += ' '.join(line) + '\n'
        with open(item.replace('tokenized_content', f'tokenized_{k}'), 'w') as f:
            f.write(tokens)


if __name__ == '__main__':

    tokenize()