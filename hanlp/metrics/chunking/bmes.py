# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-09-14 21:55

from hanlp.common.vocab import Vocab
from hanlp.metrics.chunking.f1 import ChunkingF1
from hanlp.metrics.chunking.sequence_labeling import get_entities


class BMES_F1(ChunkingF1):

    def __init__(self, tag_vocab: Vocab, from_logits=True, suffix=False, name='f1', dtype=None, **kwargs):
        super().__init__(tag_vocab, from_logits, name, dtype, **kwargs)
        self.nb_correct = 0
        self.nb_pred = 0
        self.nb_true = 0
        self.suffix = suffix

    def update_tags(self, true_tags, pred_tags):
        for t, p in zip(true_tags, pred_tags):
            self.update_entities(get_entities(t, self.suffix), get_entities(p, self.suffix))
        return self.result()

    def update_entities(self, true_entities, pred_entities):
        true_entities = set(true_entities)
        pred_entities = set(pred_entities)
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)
        self.nb_correct += nb_correct
        self.nb_pred += nb_pred
        self.nb_true += nb_true

    def result(self):
        nb_correct = self.nb_correct
        nb_pred = self.nb_pred
        nb_true = self.nb_true
        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        score = 2 * p * r / (p + r) if p + r > 0 else 0

        return score

    def reset_states(self):
        self.nb_correct = 0
        self.nb_pred = 0
        self.nb_true = 0
