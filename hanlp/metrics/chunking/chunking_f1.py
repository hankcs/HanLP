# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-11 22:14
from typing import List

from hanlp.metrics.chunking.sequence_labeling import get_entities
from hanlp.metrics.f1 import F1
from hanlp.metrics.metric import Metric


class ChunkingF1(F1):

    def __call__(self, pred_tags: List[List[str]], gold_tags: List[List[str]]):
        for p, g in zip(pred_tags, gold_tags):
            pred = set(get_entities(p))
            gold = set(get_entities(g))
            self.nb_pred += len(pred)
            self.nb_true += len(gold)
            self.nb_correct += len(pred & gold)
