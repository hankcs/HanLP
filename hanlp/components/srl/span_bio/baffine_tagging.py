# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-04 13:59
import math

import torch
from torch import nn

from hanlp.components.parsers.biaffine.biaffine import Biaffine
from hanlp.components.parsers.biaffine.mlp import MLP
from hanlp.layers.crf.crf import CRF


class BiaffineTaggingDecoder(nn.Module):

    def __init__(self,
                 n_rels,
                 hidden_size,
                 n_mlp_rel=300,
                 mlp_dropout=0.2,
                 crf=False) -> None:
        super().__init__()
        self.mlp_rel_h = MLP(n_in=hidden_size,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=hidden_size,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)
        bias = 1 / math.sqrt(self.rel_attn.weight.size(1))
        nn.init.uniform_(self.rel_attn.weight, -bias, bias)
        self.crf = CRF(n_rels) if crf else None

    # noinspection PyUnusedLocal
    def forward(self, x: torch.Tensor, **kwargs):
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        return s_rel


class SpanBIOSemanticRoleLabelingModel(nn.Module):

    def __init__(self,
                 embed,
                 encoder,
                 num_labels: int,
                 n_mlp_rel,
                 mlp_dropout,
                 crf=False,
                 ) -> None:
        super().__init__()
        self.embed = embed
        self.encoder = encoder
        hidden_size = encoder.get_output_dim() if encoder else embed.get_output_dim()
        self.decoder = BiaffineTaggingDecoder(
            num_labels,
            hidden_size,
            n_mlp_rel,
            mlp_dropout,
            crf,
        )

    def forward(self, batch, mask):
        x = self.embed(batch)
        if self.encoder:
            x = self.encoder(x, mask=mask)
        x = self.decoder(x)
        return x
