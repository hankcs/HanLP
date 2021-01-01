# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-06 14:37
from typing import Union, List

from alnlp.modules import feedforward

from hanlp.common.structure import ConfigTracker


class FeedForward(feedforward.FeedForward, ConfigTracker):
    def __init__(self, input_dim: int, num_layers: int, hidden_dims: Union[int, List[int]],
                 activations: Union[str, List[str]], dropout: Union[float, List[float]] = 0.0) -> None:
        super().__init__(input_dim, num_layers, hidden_dims, activations, dropout)
        ConfigTracker.__init__(self, locals())
