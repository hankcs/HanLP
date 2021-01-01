# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-12 17:56
from alnlp import metrics
from hanlp.metrics.metric import Metric


class CategoricalAccuracy(metrics.CategoricalAccuracy, Metric):
    @property
    def score(self):
        return self.get_metric()

    def __repr__(self) -> str:
        return f'Accuracy:{self.score:.2%}'


class BooleanAccuracy(metrics.BooleanAccuracy, CategoricalAccuracy):
    pass
