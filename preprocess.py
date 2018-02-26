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

    import numpy as np

    data = dict()
    for item in (glob('rumor/*.json') + glob('truth/*.json')):
        raw = json.load(open(item))
        span = np.array(list(map(lambda x: x['t'], raw)))
        data[item] = span - np.min(span)
    np.savez('data/prop_span.npz', **data)


if __name__ == '__main__':

    get_prop_info()