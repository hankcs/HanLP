# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-01 16:51
from typing import Any, Tuple

import torch

from hanlp.components.parsers.biaffine.biaffine_model import BiaffineDecoder
from hanlp.components.parsers.biaffine.mlp import MLP
from hanlp.components.parsers.constituency.treecrf import CRF2oDependency
from hanlp.components.parsers.second_order.affine import Triaffine


class TreeCRFDecoder(BiaffineDecoder):
    def __init__(self, hidden_size, n_mlp_arc, n_mlp_sib, n_mlp_rel, mlp_dropout, n_rels) -> None:
        super().__init__(hidden_size, n_mlp_arc, n_mlp_rel, mlp_dropout, n_rels)
        self.mlp_sib_s = MLP(hidden_size, n_mlp_sib, dropout=mlp_dropout)
        self.mlp_sib_d = MLP(hidden_size, n_mlp_sib, dropout=mlp_dropout)
        self.mlp_sib_h = MLP(hidden_size, n_mlp_sib, dropout=mlp_dropout)

        self.sib_attn = Triaffine(n_in=n_mlp_sib, bias_x=True, bias_y=True)
        self.crf = CRF2oDependency()

    def forward(self, x, mask=None, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_arc, s_rel = super(TreeCRFDecoder, self).forward(x, mask)
        sib_s = self.mlp_sib_s(x)
        sib_d = self.mlp_sib_d(x)
        sib_h = self.mlp_sib_h(x)
        # [batch_size, seq_len, seq_len, seq_len]
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        return s_arc, s_sib, s_rel
