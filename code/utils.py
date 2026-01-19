import csv
import numpy as np
from pathlib import Path
from Levenshtein import distance

import torch
from torch.utils.data import TensorDataset

import config


def reader_autoregressive(file):
    X = []
    y = []
    with open(Path(file), 'r') as data:
        reader = csv.reader(data, delimiter='\t')
        for seq, _ in reader:
            encoded_seq = [config.CHAR_TO_CODE[char] for char in seq]
            X.append(encoded_seq[len(config.LEFT_ADAPTER): -1])
            y.append(encoded_seq[len(config.LEFT_ADAPTER) + 1: ])

    X, y =  torch.LongTensor(np.array(X)), torch.LongTensor(np.array(y))

    return TensorDataset(X, y)


def reader(file):
    X = []
    y = []
    with open(Path(file), 'r') as data:
        reader = csv.reader(data, delimiter='\t')
        for seq, expr in reader:
            y.append(float(expr))
            X.append(seq)

        return np.array(X, dtype='>110U'), np.array(y, dtype=np.float32).flatten()
    

def writer(sequences, expressions, file):
    with open(Path(file), 'w') as out:
        for seq, expr in zip(sequences, expressions):
            out.write(seq + '\t' + str(expr) + '\n')


def shuffle(seq, expr):
    shuffled = np.random.choice(seq.shape[0], size=seq.shape[0], replace=False)

    return seq[shuffled], expr[shuffled]


def get_distance_matrix(seq1, seq2):
    seq1, _ = reader(seq1)
    seq2, _ = reader(seq2)
    m = np.zeros(shape=(seq1.shape[0], seq2.shape[0]), dtype=np.int32)
    for i in range(seq1.shape[0]):
        for j in range(seq2.shape[0]):
          m[i, j] = distance(seq1[i], seq2[j])

    return m