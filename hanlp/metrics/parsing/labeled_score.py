# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-27 00:49

import tensorflow as tf


class LabeledScore(object):

    def __init__(self, eps=1e-5):
        super(LabeledScore, self).__init__()

        self.eps = eps
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        return f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        arc_mask = (arc_preds == arc_golds)[mask]
        rel_mask = (rel_preds == rel_golds)[mask] & arc_mask

        self.total += len(arc_mask)
        self.correct_arcs += int(tf.math.count_nonzero(arc_mask))
        self.correct_rels += int(tf.math.count_nonzero(rel_mask))

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return self.las

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)

    def reset_states(self):
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def to_dict(self) -> dict:
        return {'UAS': self.uas, 'LAS': self.las}
