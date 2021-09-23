# -*- coding: utf8 -*-
#
import torch
from torch import nn
from torch.functional import F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.model.biaffine import Biaffine


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.3):
        super(MLP, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=n_in, out_features=n_out)

    def forward(self, x):
        y = F.relu(self.linear(x))
        return self.dropout(y)


class LSTMLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(
            n_in, n_out // 2,
            num_layers=3, bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.33)

    def forward(self, x, seq_len):
        pack = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        out, (h_0, c_0) = self.lstm(pack)
        out, output_lengths = pad_packed_sequence(out, batch_first=True)
        out = self.dropout(out)
        return out


class BiaffineDepModel(torch.nn.Module):
    def __init__(self, num_embeddings, n_labels: int = None):
        super(BiaffineDepModel, self).__init__()

        self.embed = nn.Embedding(num_embeddings, 100)
        self.lstm = LSTMLayer(100, 400)
        self.start_layer = MLP(n_in=400, n_out=500, dropout=0.33)
        self.end_layer = MLP(n_in=400, n_out=500, dropout=0.33)

        self.biaffine = Biaffine(n_in=500, n_out=n_labels)

    def forward(self, words):
        seq_len = words.not_equal(0).sum(1).to('cpu')
        out = self.embed(words)
        out = self.lstm(out, seq_len)
        start_out = self.start_layer(out)
        end_out = self.end_layer(out)

        out = self.biaffine(x=start_out, y=end_out)
        out = out.permute(0, 2, 3, 1).contiguous()
        return F.log_softmax(out, dim=-1)
