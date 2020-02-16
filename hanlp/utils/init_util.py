# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-27 13:25
import math

import torch
from torch import nn
import functools


def embedding_uniform(tensor:torch.Tensor, seed=233):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        fan_out = tensor.size(-1)
        bound = math.sqrt(3.0 / fan_out)
        return tensor.uniform_(-bound, bound, generator=gen)
