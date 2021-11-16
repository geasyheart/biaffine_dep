# -*- coding: utf8 -*-
#
import math

import torch.nn

from src.model.biaffine import Biaffine
from src.model.mlp import MLP
from src.model.transformer import TransformerEmbedding


class BiaffineDepModel(torch.nn.Module):
    def __init__(self, transformer: str, n_arc_mlp=500, n_rel_mlp=100, n_out: int=None):
        super(BiaffineDepModel, self).__init__()

        self.encoder = TransformerEmbedding(model=transformer, n_layers=4, dropout=0.33)
        self.arc_mlp_d = MLP(n_in=self.encoder.n_out, n_out=n_arc_mlp, dropout=0.33)
        self.arc_mlp_h = MLP(n_in=self.encoder.n_out, n_out=n_arc_mlp, dropout=0.33)

        self.rel_mlp_d = MLP(n_in=self.encoder.n_out, n_out=n_rel_mlp, dropout=0.33)
        self.rel_mlp_h = MLP(n_in=self.encoder.n_out, n_out=n_rel_mlp, dropout=0.33)

        self.arc_attn = Biaffine(n_in=n_arc_mlp, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_rel_mlp,n_out=n_out, bias_x=True, bias_y=True)

    def forward(self, subwords):
        bert_out = self.encoder(subwords=subwords)
        mask = subwords.ne(self.encoder.tokenizer.pad_token_id).any(-1)

        arc_d = self.arc_mlp_d(bert_out)
        arc_h = self.arc_mlp_h(bert_out)
        rel_d = self.rel_mlp_d(bert_out)
        rel_h = self.rel_mlp_h(bert_out)
        # 再次更新，此次有两个点可以注意的哦
        # 1. 在计算s_arc的时候,mask在第0个位置参与了模型计算，只是在计算loss的时候才忽略掉了，所以这是一个小改动点
        # 应更改成如下：
        # mask[:, 0] = 0
        # self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), -math.inf)

        # 2. 关于s_arc计算，masked_fill_这里不方便理解，换种方式会更容易
        arc_v = self.arc_attn(arc_d, arc_h)
        s_arc = arc_v.masked_fill_(~mask.unsqueeze(1), -math.inf)

        # friendly_s_arc = arc_v.masked_fill_(~(mask.unsqueeze(1) & mask.unsqueeze(2)), -math.inf)
        # assert (s_arc == friendly_s_arc).all()

        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        return s_arc, s_rel
