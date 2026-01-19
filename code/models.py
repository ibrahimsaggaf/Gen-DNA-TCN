'''
This code is modified based on the impelemtation by
https://github.com/locuslab/TCN
'''

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        
        self.chomp_size = chomp_size


    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dropout = dropout
        
        self.relu = nn.ReLU()
        self.temp_block = nn.Sequential(
            weight_norm(nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation)),
            Chomp1d(self.padding),
            self.relu,
            nn.Dropout(self.dropout),
            weight_norm(nn.Conv1d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation)),
            Chomp1d(self.padding),
            self.relu,
            nn.Dropout(self.dropout)
        )
        self.downsample = nn.Conv1d(self.in_channels, self.out_channels, 1) if self.in_channels != self.out_channels else None
        

    def forward(self, x):
        out = self.temp_block(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class DNATCN(nn.Module):
    def __init__(self, em_dim, output_size, channel_size, max_seq_len, kernel_size, stride, 
                 dropout, em_dropout, num_char, predictor):
        super().__init__()
        self.predictor = predictor
        
        # Network depth is defined based on the longest sequence       
        levels = int(math.log(max_seq_len / (kernel_size - 1), 2))
        out_channels = [channel_size] * levels
        layers = []
        for i in range(levels):
            dilation = 2 ** i
            in_c = em_dim if i == 0 else out_channels[i - 1]
            out_c = out_channels[i]
            layers += [TemporalBlock(in_c, out_c, kernel_size, stride, dilation,
                                     padding=(kernel_size - 1) * dilation, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.em = nn.Embedding(num_char, em_dim)
        self.em_dropout = nn.Dropout(em_dropout)

        if self.predictor:
            self.linear = nn.Linear(channel_size, output_size)


    def forward(self, x):
        x = self.em_dropout(self.em(x))
        x = self.tcn(x.transpose(1, 2))

        if self.predictor:
            x = self.linear(x[:, :, -1])
            return x.view(-1)

        else:
            return x.transpose(1, 2).contiguous()


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)

        return x


def weights_init(net):
    if type(net) == nn.Conv1d:
        net.weight.data.normal_(0.0, 0.02)

    if type(net) == nn.Embedding:
        net.weight.data.uniform_(-0.1, 0.1)

    if type(net) == nn.Linear:
        net.weight.data.normal_(0.0, 0.02)


def load_tcn(em_dim, output_size, channel_size, max_seq_len, kernel_size, stride, dropout,
               em_dropout, num_char, mode, model=None, pre_checkpoint=None, gen_checkpoint=None,
               clf_checkpoint=None):
    
    if mode == 'training':
        gen_tcn = DNATCN(em_dim, output_size, channel_size, max_seq_len, kernel_size, stride, dropout, 
                        em_dropout, num_char, predictor=False)
        clf = LinearClassifier(in_dim=channel_size, out_dim=num_char)
        clf.apply(weights_init)
    
        if model in [1, 2]:
            checkpoint = torch.load(Path(pre_checkpoint), map_location=torch.device('cpu'))

            # Remove the weights of the linear head in Pre-DNA-TCN
            del checkpoint['linear.weight']
            del checkpoint['linear.bias']

            # Initialise Gen-DNA-TCN with Pre-DNA-TCN weights
            gen_tcn.load_state_dict(checkpoint)

        elif model == 3:
            gen_tcn.apply(weights_init)

        return gen_tcn.train(), clf.train()
    
    if mode == 'inference':
        pre_tcn = DNATCN(em_dim, output_size, channel_size, max_seq_len, kernel_size, stride, dropout, 
                        em_dropout, num_char, predictor=True)
        checkpoint = torch.load(Path(pre_checkpoint), map_location=torch.device('cpu'))
        pre_tcn.load_state_dict(checkpoint)

        gen_tcn = DNATCN(em_dim, output_size, channel_size, max_seq_len, kernel_size, stride, dropout, 
                        em_dropout, num_char, predictor=False)
        checkpoint = torch.load(Path(gen_checkpoint), map_location=torch.device('cpu'))
        gen_tcn.load_state_dict(checkpoint)

        clf = LinearClassifier(in_dim=channel_size, out_dim=num_char)
        checkpoint = torch.load(Path(clf_checkpoint), map_location=torch.device('cpu'))
        clf.load_state_dict(checkpoint)

        return pre_tcn.eval(), gen_tcn.eval(), clf.eval()