# MIT License
#
# Copyright (c) 2020 Yu Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import Counter

from hanlp.metrics.metric import Metric


class SpanMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()
        self.reset(eps)

    # noinspection PyAttributeOutsideInit
    def reset(self, eps=1e-12):
        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            upred = Counter([(i, j) for i, j, label in pred])
            ugold = Counter([(i, j) for i, j, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)
        return self

    def __repr__(self):
        s = f"UCM: {self.ucm:.2%} LCM: {self.lcm:.2%} "
        s += f"UP: {self.up:.2%} UR: {self.ur:.2%} UF: {self.uf:.2%} "
        s += f"LP: {self.lp:.2%} LR: {self.lr:.2%} LF: {self.lf:.2%}"

        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)
