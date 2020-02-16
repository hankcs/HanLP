# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-03 11:35
from abc import ABC, abstractmethod


class Metric(ABC):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __eq__(self, other):
        return self.score == other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    def __ne__(self, other):
        return self.score != other

    @property
    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def __call__(self, pred, gold, mask=None):
        pass

    def __repr__(self) -> str:
        return f'{self.score}:.4f'

    def __float__(self):
        return self.score

    @abstractmethod
    def reset(self):
        pass
