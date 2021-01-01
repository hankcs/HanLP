# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-01 15:28
from torch import nn


# noinspection PyAbstractClass
class DependencyModel(nn.Module):
    def __init__(self, embed: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.embed = embed
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch, mask):
        x = self.embed(batch, mask=mask)
        x = self.encoder(x, mask)
        return self.decoder(x, mask=mask)
