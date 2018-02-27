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


def save_tokens(tokens, fname):

    with open(fname.replace('cutted', 'tokenized'), 'w') as f:
        for line in tokens:
            f.write(' '.join(map(str, line)) + '\n')
    print(f'saved {fname}')


def tokenize():

    from multiprocessing.pool import Pool
    pool = Pool(processes=3)
    word_dict, idx_dict = {}, []
    vocab_size = 0
    os.mkdir('data/tokenized_content')
    for item in glob('data/cutted_content/*.txt'):
        tokens = []
        for line in open(item).readlines():
            tokens.append([])
            for word in line.split():
                if word not in word_dict:
                    word_dict[word] = vocab_size
                    idx_dict.append(word)
                    vocab_size += 1
                tokens[-1].append(word_dict[word])
        print(f'tokenized {item}')
        pool.apply_async(save_tokens, (tokens, item))
    pool.close()
    pool.join()


if __name__ == '__main__':

    tokenize()