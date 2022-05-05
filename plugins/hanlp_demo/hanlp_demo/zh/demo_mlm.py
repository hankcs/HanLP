# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-01-29 21:11
from hanlp.components.lm.mlm import MaskedLanguageModel

mlm = MaskedLanguageModel()
mlm.load('bert-base-chinese')
print(mlm('生活的真谛是[MASK]。'))

# Batching is always faster
print(mlm(['生活的真谛是[MASK]。', '巴黎是[MASK][MASK]的首都。']))
