# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-08 17:56
from alnlp.modules.pass_through_encoder import PassThroughEncoder as _PassThroughEncoder

from hanlp.common.structure import ConfigTracker


class PassThroughEncoder(_PassThroughEncoder, ConfigTracker):
    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim)
        ConfigTracker.__init__(self, locals())
