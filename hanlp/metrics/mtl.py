# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-03 00:16
from hanlp.metrics.metric import Metric


class MetricDict(Metric, dict):
    _COLORS = ["magenta", "cyan", "green", "yellow"]

    @property
    def score(self):
        return sum(float(x) for x in self.values()) / len(self)

    def __call__(self, pred, gold):
        for metric in self.values():
            metric(pred, gold)

    def reset(self):
        for metric in self.values():
            metric.reset()

    def __repr__(self) -> str:
        return ' '.join(f'({k} {v})' for k, v in self.items())

    def cstr(self, idx=None, level=0) -> str:
        if idx is None:
            idx = [0]
        prefix = ''
        for _, (k, v) in enumerate(self.items()):
            color = self._COLORS[idx[0] % len(self._COLORS)]
            idx[0] += 1
            child_is_dict = isinstance(v, MetricDict)
            _level = min(level, 2)
            # if level != 0 and not child_is_dict:
            #     _level = 2
            lb = '{[('
            rb = '}])'
            k = f'[bold][underline]{k}[/underline][/bold]'
            prefix += f'[{color}]{lb[_level]}{k} [/{color}]'
            if child_is_dict:
                prefix += v.cstr(idx, level + 1)
            else:
                prefix += f'[{color}]{v}[/{color}]'
            prefix += f'[{color}]{rb[_level]}[/{color}]'
        return prefix
