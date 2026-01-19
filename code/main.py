import torch
import argparse
import numpy as np
from pathlib import Path

import config
from train import fit
from utils import writer, get_distance_matrix
from models import load_tcn
from sample import bin_based_sampling, random_sampling
from generate import generate_sequences, predict, remove_identical


def main(args):
    if args.mode == 'training':
        device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

        if args.model in [1, 2]:
            gen_tcn, clf = load_tcn(
                args.em_dim, args.out_size, args.c_size, args.max_seq_len, args.k_size, args.stride, args.dropout,
                args.em_dropout, config.NUM_CHAR, args.mode, args.model, pre_checkpoint=args.pre_checkpoint
            )

        elif args.model == 3:
            gen_tcn, clf = load_tcn(
                args.em_dim, args.out_size, args.c_size, args.max_seq_len, args.k_size, args.stride, args.dropout,
                args.em_dropout, config.NUM_CHAR, args.mode, args.model
            )

        else:
            raise NotImplementedError(
            f'The model {args.model} is not implemeneted. Please choose one of the following values:'\
            ' 1, 2 or 3.'
            )
        
        fit(
            args.file_train_seq, args.file_val_seq, gen_tcn, clf, args.b_size, args.lr, args.epochs, args.hist_len, args.clip, 
            device, args.model, args.seed, args.res_path
        )

    elif args.mode == 'inference':
        device = torch.device('cpu')
        pre_tcn, gen_tcn, clf = load_tcn(
            args.em_dim, args.out_size, args.c_size, args.max_seq_len, args.k_size, args.stride, args.dropout,
            args.em_dropout, config.NUM_CHAR, args.mode, args.model, pre_checkpoint=args.pre_checkpoint,
            gen_checkpoint=args.gen_checkpoint, clf_checkpoint=args.clf_checkpoint
        )

        if args.init_type == 1:
            for n in range(args.min_n, args.max_n + 1):
                gen_seq = generate_sequences(
                    gen_tcn, clf, n, args.b_size, args.max_seq_len, device, args.init_type, file_train_seq=args.file_train_seq
                )
                gen_seq = remove_identical(gen_seq, args.sequences_71k, args.sequences_6m)
                expressions = predict(gen_seq, pre_tcn, args.b_size, device)
                writer(
                    gen_seq, expressions, 
                    Path(args.res_path, f'Model_{args.model}_generated_DNA_sequences_using_training_init_{n}.txt')
                )

            sampled_gen_seq, sampled_expressions = bin_based_sampling(
                args.model, args.sequences_71k, args.min_n, args.max_n, args.res_path
            )
            writer(
                sampled_gen_seq, sampled_expressions, 
                Path(args.res_path, f'Model_{args.model}_generated_DNA_sequences_bin_based_sampling.txt')
            )

        elif args.init_type == 2:
            gen_seq = generate_sequences(
                gen_tcn, clf, args.n, args.b_size, args.max_seq_len, device, args.init_type
            )
            gen_seq = remove_identical(gen_seq, args.sequences_71k, args.sequences_6m)
            expressions = predict(gen_seq, pre_tcn, args.b_size, device)
            writer(
                gen_seq, expressions, 
                Path(args.res_path, f'Model_{args.model}_generated_DNA_sequences_permutation.txt')
            )

            sampled_gen_seq, sampled_expressions = random_sampling(gen_seq, expressions, args.sample_size)
            writer(
                sampled_gen_seq, sampled_expressions, 
                Path(args.res_path, f'Model_{args.model}_generated_DNA_sequences_permutation_random_sampling.txt')
            )

        else:
            raise NotImplementedError(
            f'The type of starting nucleotides {args.init_type} is not implemeneted. Please choose one of the following values:'\
            ' 1 or 2.'
            )

    elif args.mode == 'distance':
        dist_matrix = get_distance_matrix(args.seq1, args.seq2)
        np.save(Path(args.res_path, f'dist_matrix.npy', dist_matrix))

    else:
        raise NotImplementedError(
            f'The mode {args.mode} is not implemeneted. Please choose one of the following values:'\
            ' training or inference.'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=int, help='the varient of the Gen-DNA-TCN model. Please choose one of the following values: 1, 2 or 3.')
    parser.add_argument('--mode', help='To train the Gen-DNA-TCN model choose "training",'\
                        ' to generate synthetic yeast promoter DNA sequences choose "inference",'\
                        ' to measure pair-wise distances between real and synthetic sequences choose "distance".'
    )
    parser.add_argument('--init_type', required=False, type=int, help='type of starting nucleotides.'\
                        ' 1 denotes starting nucleotides are extracted from the real 56,879 training sequences,'\
                        ' whilst 2 denotes starting nucleotides are made by random permutations.'
    )

    parser.add_argument('--file_train_seq', required=False, default='training_sequences.txt', help='the full apth for the training sequences file.')
    parser.add_argument('--file_val_seq', required=False, default='validation_sequences.txt', help='the full path for the validation sequences file.')
    parser.add_argument('--sequences_71k', required=False, default='sequences_71k.txt', help='the full path for the 71K real sequences file.')
    parser.add_argument('--sequences_6m', required=False, default='sequences_6m.txt', help='the full path for the 6M real sequences file.')
    parser.add_argument('--seq1', required=False, help='the full path for the first set of sequences to compute the distances matrix.')
    parser.add_argument('--seq2', required=False, help='the full path for the second set of sequences to compute the distances matrix.')

    parser.add_argument('--pre_checkpoint', required=False, help='the full path for the Pre-DNA-TCN checkpoint.')
    parser.add_argument('--gen_checkpoint', required=False, help='the full path for the Gen-DNA-TCN checkpoint.')
    parser.add_argument('--clf_checkpoint', required=False, help='the full path for the Gen-DNA-TCN classification head checkpoint.')
   
    parser.add_argument('--max_seq_len', required=False, type=int, help='max sequence length.')
    parser.add_argument('--n', required=False, type=int, help='the length of starting nucleotides.')
    parser.add_argument('--min_n', required=False, type=int, help='the minimum length of starting nucleotides.')
    parser.add_argument('--max_n', required=False, type=int, help='the maximum length of starting nucleotides.')
    parser.add_argument('--b_size', required=False, type=int, help='batch size.')
    parser.add_argument('--lr', required=False, type=float, help='learning rate.')
    parser.add_argument('--epochs', required=False, type=int, help='number of epochs.')
    parser.add_argument('--hist_len', required=False, type=int, help='effective history length.')
    parser.add_argument('--clip', required=False, type=float, help='gradient clip.')
    parser.add_argument('--em_dim', required=False, type=int, help='embeddings dimension.')
    parser.add_argument('--out_size', required=False, type=int, help='The Pre-DNA-TCN output size.')
    parser.add_argument('--c_size', required=False, type=int, help='channel size.')
    parser.add_argument('--k_size', required=False, type=int, help='kernel size.')
    parser.add_argument('--stride', required=False, type=int, help='stride size.')
    parser.add_argument('--dropout', required=False, type=float, help='TCN dropout.')
    parser.add_argument('--em_dropout', required=False, type=float, help='embeddings dropout.')
    parser.add_argument('--device', required=False, type=int, help='the device for training')
    parser.add_argument('--seed', required=False, type=int, help='seed') 
    parser.add_argument('--res_path', required=False, help='the path where the results/output files are saved.')  

    args = parser.parse_args()
    main(args)

    