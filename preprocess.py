#!/usr/bin/env python3.6
import json
import os
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
    from collections import defaultdict

    data = {}
    for item in (glob('rumor/*.json') + glob('truth/*.json')):
        raw = sorted(json.load(open(item)), key=lambda x: x['t'])
        begin = raw[0]['t']
        layer = defaultdict(int)
        graph = []
        for msg in raw:
            x, y, t = msg['parent'], msg['mid'], msg['t']
            x = 0 if x == None else x
            layer[y] = layer[x] + 1
            graph.append((t - begin, layer[x], layer[y], int(x), int(y)))
        data[item] = np.vstack(graph)
    np.savez('data/prop_graph.npz', **data)


def get_content():

    import os, thulac

    def cut_content():

        thul = thulac.thulac(seg_only=True)
        os.mkdir('data/original_content_')
        os.mkdir('data/cutted_content_')

        for item in (glob('rumor/*.json') + glob('truth/*.json')):
            fid = os.path.basename(item)[:-5]
            original = f'data/original_content_/{fid}.txt'
            cutted = f'data/cutted_content_/{fid}.txt'
            raw = sorted(json.load(open(item)), key=lambda x: x['t'])
            begin = raw[0]['t']
            data = list(map(lambda x: x['text'] + '\n', raw))
            with open(original, 'w') as f:
                f.writelines(data)
            thul.cut_f(original, cutted)
                
    ddl = [1, 2, 6, 12, 24, 36, 48, 72, 96]
    # for i in ddl:
    #     os.mkdir(f'data/cutted_content_{i}')
    for item in (glob('rumor/*.json') + glob('truth/*.json')):
        fid = os.path.basename(item)[:-5]
        raw = sorted(json.load(open(item)), key=lambda x: x['t'])
        begin = raw[0]['t']
        for dtime in ddl:
            for i in range(len(raw)):
                if raw[i]['t'] - begin > dtime * 3600:
                    os.system(f'head -{i} data/cutted_content_/{fid}.txt > data/cutted_content_{dtime}/{fid}.txt')
                    break
            else:
                os.system(f'cat data/cutted_content_/{fid}.txt > data/cutted_content_{dtime}/{fid}.txt')



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

    get_content()
