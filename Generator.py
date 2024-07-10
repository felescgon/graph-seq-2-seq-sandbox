import torch.nn as nn
import torch
from torch.nn.utils import rnn as rnn_utils
import numpy as np

class Generator(nn.Module):
    def __init__(self, sequence_length, n_features, nz=100, ngf=64):
        super(Generator, self).__init__()
        #input size: sequence_length * n_features * z_dimension_expansor
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose1d(ngf, n_features, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, X):
        #N, 1, 1
        out = self.main(X).permute(0, 2, 1)

        return out
        #return self.linear(out)