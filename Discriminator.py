import torch.nn as nn
import torch
from torch.nn.utils import rnn as rnn_utils

class Discriminator(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs, rnn_layer=nn.LSTM, batch_norm = True, **kwargs):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None
        self.cell = None
        self.basic_rnn = rnn_layer(self.n_features, self.hidden_dim, batch_first=True, **kwargs)
        self.bn_hidden = nn.BatchNorm1d(hidden_dim)
        self.bn_x = None
        self.bn_x = None
        if batch_norm:
            self.bn_x = nn.BatchNorm1d(n_features)

        output_dim = (self.basic_rnn.bidirectional + 1) * self.hidden_dim
        # Classifier to produce as many logits as outputs
        self.classifier = nn.Linear(output_dim, self.n_outputs)

    def forward(self, X):
        is_packed = isinstance(X, nn.utils.rnn.PackedSequence)
        # X is a PACKED sequence, there is no need to permute
        if self.bn_x is not None:
            X_permuted = X.permute(0, 2, 1)
            X_normalised = self.bn_x(X_permuted)
            X = X_normalised.permute(0, 2, 1)

        rnn_out, self.hidden = self.basic_rnn(X)

        if self.bn_x is not None:
            rnn_out_permuted = rnn_out.permute(0, 2, 1)
            rnn_out_normalised = self.bn_hidden(rnn_out_permuted)
            rnn_out = rnn_out_normalised.permute(0, 2, 1)

        if isinstance(self.basic_rnn, nn.LSTM):
            self.hidden, self.cell = self.hidden

        if is_packed:
            # unpack the output
            batch_first_output, seq_sizes = rnn_utils.pad_packed_sequence(rnn_out, batch_first=True)
            seq_slice = torch.arange(seq_sizes.size(0))
        else:
            batch_first_output = rnn_out
            seq_sizes = 0  # so it is -1 as the last output
            seq_slice = slice(None, None, None)  # same as ':'

        # only last item in sequence (N, 1, H)
        last_output = batch_first_output[seq_slice, seq_sizes - 1]

        # classifier will output (N, 1, n_outputs)
        out = self.classifier(last_output)

        # final output is (N, n_outputs)
        return torch.sigmoid(out.view(-1, self.n_outputs))