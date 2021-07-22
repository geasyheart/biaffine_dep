# -*- coding: utf8 -*-
#
import torch.nn

from src.model.biaffine import Biaffine
from src.model.mlp import MLP
from src.model.transformer import TransformerEmbedding


class BiaffineDepModel(torch.nn.Module):
    def __init__(self, transformer: str, hidden_size=300, n_labels: int = None):
        super(BiaffineDepModel, self).__init__()

        self.encoder = TransformerEmbedding(model=transformer, n_layers=4, dropout=0.33)
        self.start_layer = MLP(n_in=self.encoder.n_out, n_out=hidden_size, dropout=0.33)
        self.end_layer = MLP(n_in=self.encoder.n_out, n_out=hidden_size, dropout=0.33)

        self.biaffine = Biaffine(n_in=hidden_size, n_out=n_labels)

    def forward(self, subwords):
        bert_out = self.encoder(subwords=subwords)

        start_out = self.start_layer(bert_out)
        end_out = self.end_layer(bert_out)

        out = self.biaffine(x=start_out, y=end_out)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out
