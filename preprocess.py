#!/usr/bin/env python3
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

    data = dict()
    for item in (glob('rumor/*.json') + glob('truth/*.json')):
        raw = json.load(open(item))
        begin = raw[0]['t']
        span = list(map(lambda x: max(0, x['t'] - begin) + 1, raw))
        data[item] = span

    with open('data/prop_span.json', 'w') as fw:
        json.dump(data, fw)


if __name__ == '__main__':

    get_prop_info()