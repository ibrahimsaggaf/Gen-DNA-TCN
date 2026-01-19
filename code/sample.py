import numpy as np
import pandas as pd
from pathlib import Path

from utils import reader, shuffle


def get_bins_range(expression, num_bin):
    bins = pd.cut(expression, bins=num_bin, labels=[f'bin{i}' for i in range(1, num_bin + 1)])
    df = pd.DataFrame({'bin': bins, 'expr': expression})

    bins_range = []
    for bin in [f'bin{i}' for i in range(1, num_bin + 1)]:
        subset = df[df['bin'] == bin].copy()
        s = subset['expr'].min()
        e = subset['expr'].max()
        bins_range.append([s, e])

    bins_range[0][1] = bins_range[1][0]

    return bins_range


def get_mask_bin(expressions, start, end):

    return np.logical_and(expressions >= start, expressions <= end)


def bin_based_sampling(model, sequences_71k, min_n, max_n, res_path):
    _, real_expr = reader(sequences_71k)
    bins_range = get_bins_range(real_expr)
    gen_seq = np.array([], dtype='>110U')
    gen_expr = np.array([], dtype=np.float32)
    for n in range(min_n, max_n + 1):
        seq, expr = reader(Path(res_path, f'Model_{model}_generated_DNA_sequences_training_init_{n}.txt'))
        mask = np.logical_not(np.isin(seq, gen_seq))
        gen_seq = np.hstack((gen_seq, seq[mask]))
        gen_expr = np.hstack((gen_expr, expr[mask]))

    gen_seq, gen_expr = shuffle(gen_seq, gen_expr)
    sampled_seq, sampled_expr = np.array([], dtype='>110U'), np.array([], dtype=np.float32)

    for bin in bins_range:
        mask = get_mask_bin(gen_expr, bin[0], bin[1])
        sampled_seq = np.hstack((sampled_seq, gen_seq[mask]))
        sampled_expr = np.hstack((sampled_expr, gen_expr[mask]))

    return sampled_seq, sampled_expr


def random_sampling(gen_seq, expressions, size):
    sampled = np.random.choice(gen_seq.shape[0], size=size, replace=False)

    return gen_seq[sampled], expressions[sampled]

