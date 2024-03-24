import gzip
import pickle
import yaml
import numpy as np
import random
from definition import *


# 加载Phonexi-2014T数据集
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return dict(loaded_object)


def config():
    with open('./config.yaml', 'r') as file:
        conf = yaml.safe_load(file)
    return conf


config = config()


def data_augmentation(resize=(320, 240), crop_size=224, is_train=True):
    if is_train:
        left, top = np.random.randint(0, resize[0] - crop_size), np.random.randint(0, resize[1] - crop_size)
    else:
        left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2

    return (left, top, left + crop_size, top + crop_size), resize


def sampler_func(clip, sn, random_choice=True):
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                  range(int(n * i / sn),
                                                                                        max(int(n * i / sn) + 1,
                                                                                            int(n * (
                                                                                                    i + 1) / sn))))
                       for i in range(sn)]
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                max(int(
                                                                                                    n * i / sn) + 1,
                                                                                                    int(n * (
                                                                                                            i + 1) / sn))))
                       for i in range(sn)]
    return f(clip)


def NoiseInjecting(raw_gloss, noise_rate=0.15, noise_type='omit_last', random_shuffle=False, is_train=True):
    new_gloss = []

    for ii, gloss in enumerate(raw_gloss):
        text = gloss.split()

        if noise_type == 'omit':
            # del noise
            if random.uniform(0, 1) <= 1. and is_train:
                index = sampler_func(len(text), int(len(text) * (1. - noise_rate)), random_choice=is_train)
                noise_gloss = []
                noise_idx = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
                        noise_idx.append(i)
            else:
                noise_gloss = [d for d in text]

        elif noise_type == 'omit_last':
            if random.uniform(0, 1) <= 1.0 and is_train:
                index = np.arange(0, len(text) - int(np.ceil(len(text) * (np.random.uniform(0, noise_rate, (1,))))), 1,
                                  dtype=int)
                noise_gloss = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
            else:
                noise_gloss = [d for d in text]

        if is_train and random_shuffle and random.uniform(0, 1) > 0.5:
            random.shuffle(noise_gloss)  # random shuffle sequence

        new_gloss.append(' '.join(noise_gloss))
    return new_gloss
