# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-11 18:37
from hanlp.datasets.ner.msra import MSRA_NER_TRAIN

from hanlp.components.taggers.transformers.transformer_transform import TransformerTransform

transform = TransformerTransform(max_seq_length=128)

for x, y in transform.file_to_inputs(MSRA_NER_TRAIN):
    assert len(x) == len(y)
    if not len(x) or len(x) > 126:
        print(x)
