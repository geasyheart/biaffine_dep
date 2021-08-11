# -*- coding: utf8 -*-
#
import math

import torch.nn

from src.model.biaffine import Biaffine
from src.model.mlp import MLP
from src.model.transformer import TransformerEmbedding


class BiaffineDepModel(torch.nn.Module):
    def __init__(self, transformer: str, n_arc_mlp=500, n_rel_mlp=100):
        super(BiaffineDepModel, self).__init__()

        self.encoder = TransformerEmbedding(model=transformer, n_layers=4, dropout=0.33)
        self.arc_mlp_d = MLP(n_in=self.encoder.n_out, n_out=n_arc_mlp, dropout=0.33)
        self.arc_mlp_h = MLP(n_in=self.encoder.n_out, n_out=n_arc_mlp, dropout=0.33)

        self.rel_mlp_d = MLP(n_in=self.encoder.n_out, n_out=n_rel_mlp, dropout=0.33)
        self.rel_mlp_h = MLP(n_in=self.encoder.n_out, n_out=n_rel_mlp, dropout=0.33)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, bias_x=True, bias_y=True)

    def forward(self, subwords):
        bert_out = self.encoder(subwords=subwords)
        mask = subwords.ne(self.encoder.tokenizer.pad_token_id).any(-1)

        arc_d = self.arc_mlp_d(bert_out)
        arc_h = self.arc_mlp_h(bert_out)
        rel_d = self.rel_mlp_d(bert_out)
        rel_h = self.rel_mlp_h(bert_out)

        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), -math.inf)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel
