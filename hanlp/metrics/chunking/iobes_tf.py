# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-09-14 21:55

from hanlp.common.vocab_tf import VocabTF
from hanlp.metrics.chunking.conlleval import SpanF1
from hanlp.metrics.chunking.chunking_f1_tf import ChunkingF1_TF


class IOBES_F1_TF(ChunkingF1_TF):

    def __init__(self, tag_vocab: VocabTF, from_logits=True, name='f1', dtype=None, **kwargs):
        super().__init__(tag_vocab, from_logits, name, dtype, **kwargs)
        self.state = SpanF1()

    def update_tags(self, true_tags, pred_tags):
        # true_tags = list(itertools.chain.from_iterable(true_tags))
        # pred_tags = list(itertools.chain.from_iterable(pred_tags))
        # self.state.update_state(true_tags, pred_tags)
        for gold, pred in zip(true_tags, pred_tags):
            self.state.update_state(gold, pred)
        return self.result()

    def result(self):
        return self.state.result(full=False, verbose=False).fscore

    def reset_states(self):
        self.state.reset_state()
