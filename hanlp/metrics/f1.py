# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-10 14:55
from abc import ABC

from hanlp.metrics.metric import Metric


class F1(Metric, ABC):
    def __init__(self, nb_pred=0, nb_true=0, nb_correct=0) -> None:
        super().__init__()
        self.nb_correct = nb_correct
        self.nb_pred = nb_pred
        self.nb_true = nb_true

    def __repr__(self) -> str:
        p, r, f = self.prf
        return f"P: {p:.2%} R: {r:.2%} F1: {f:.2%}"

    @property
    def prf(self):
        nb_correct = self.nb_correct
        nb_pred = self.nb_pred
        nb_true = self.nb_true
        p = nb_correct / nb_pred if nb_pred > 0 else .0
        r = nb_correct / nb_true if nb_true > 0 else .0
        f = 2 * p * r / (p + r) if p + r > 0 else .0
        return p, r, f

    @property
    def score(self):
        return self.prf[-1]

    def reset(self):
        self.nb_correct = 0
        self.nb_pred = 0
        self.nb_true = 0

    def __call__(self, pred: set, gold: set):
        self.nb_correct += len(pred & gold)
        self.nb_pred += len(pred)
        self.nb_true += len(gold)


class F1_(Metric):
    def __init__(self, p, r, f) -> None:
        super().__init__()
        self.f = f
        self.r = r
        self.p = p

    @property
    def score(self):
        return self.f

    def __call__(self, pred, gold):
        raise NotImplementedError()

    def reset(self):
        self.f = self.r = self.p = 0

    def __repr__(self) -> str:
        p, r, f = self.p, self.r, self.f
        return f"P: {p:.2%} R: {r:.2%} F1: {f:.2%}"
