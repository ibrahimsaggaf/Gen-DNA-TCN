import numpy as np
from itertools import product

import torch
from torch.utils.data import DataLoader

import config
from utils import reader


def init_seq_permutation(n):
    init_seq = []
    for seq in product('ACGT', repeat=n):
        seq = [config.CHAR_TO_CODE[char] for char in seq]
        seq = torch.LongTensor(seq).view(1, -1)
        init_seq.append(seq)

    return torch.vstack(init_seq)


def init_seq_training(file, n):
    start = len(config.LEFT_ADAPTER)
    train_seq, _ = reader(file)
    init_seq = []
    for seq in train_seq:
        init_seq.append(seq[start: start + n])

    init_seq = set(init_seq)
    init_seq_ = []
    for seq in init_seq:
        seq = [config.CHAR_TO_CODE[char] for char in seq]
        seq = torch.LongTensor(seq).view(1, -1)
        init_seq_.append(seq)

    return torch.vstack(init_seq_)


def remove_identical(sequences, sequences_71k, sequences_6m):
    sequences = set(sequences)
    match = list(sequences & set(sequences_71k))
    sequences = sequences - set(match)

    match = list(sequences & set(sequences_6m))
    sequences = sequences - set(match)

    return np.array(sequences, dtype='>110U')


def predict(sequences, pred_tcn, batch_size, device):
    sequences_ = []
    for seq in sequences:
        seq = [config.CHAR_TO_CODE[char] for char in seq]
        seq = torch.LongTensor(seq).view(1, -1)
        sequences_.append(seq)

    sequences_ = torch.vstack(sequences_)
    data_loader = DataLoader(sequences_, batch_size=batch_size, drop_last=False, shuffle=False)
    expressions = []
    with torch.no_grad():
        for seq in data_loader:
            seq = seq.to(device)
            expr = pred_tcn(seq)
            expressions.append(expr)

    expressions = torch.hstack(expressions)

    return expressions.detach().numpy().flatten()


def generate_sequences(gen_tcn, clf, n, batch_size, seq_len, device, init_type, file_train_seq=None):
    gen_seq = []
    if init_type == 1:
        init_seq = init_seq_training(file_train_seq, n)

    elif init_type == 2:
        init_seq = init_seq_permutation(n)

    else:
        raise NotImplementedError(
            f'The type {init_type} is not implemeneted. Please enter one of the following values: '\
            '1 or 2.'
        )

    loader = DataLoader(init_seq, batch_size=batch_size, drop_last=False, shuffle=True)
    with torch.no_grad():
        for seq in loader:
            while seq.size(1) < seq_len - (len(config.LEFT_ADAPTER) + len(config.RIGHT_ADAPTER)):
                seq = seq.to(device)
                out = clf(gen_tcn(seq))
                out = torch.argmax(out.detach(), dim=-1)
                out = out[:, -1].view(-1, 1)
                seq = torch.hstack([seq, out])

            left_adapter = torch.LongTensor(config.LEFT_ADAPTER).repeat(seq.size(0), 1).to(device)
            right_adapter = torch.LongTensor(config.RIGHT_ADAPTER).repeat(seq.size(0), 1).to(device)
            seq = torch.hstack([left_adapter, seq, right_adapter])
            gen_seq.append(seq)

    gen_seq = torch.vstack(gen_seq).detach()
    gen_seq_ = []
    for i in range(gen_seq.size(0)):
        gen_seq_.append(''.join([config.CODE_TO_CHAR[code.item()] for code in gen_seq[i]]))

    return np.array(gen_seq_, dtype='>110U')