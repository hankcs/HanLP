__author__ = 'max'

import re
import numpy as np

from hanlp.metrics.metric import Metric


def is_uni_punctuation(word):
    match = re.match("^[^\w\s]+$]", word, flags=re.UNICODE)
    return match is not None


def is_punctuation(word, pos, punct_set=None):
    if punct_set is None:
        # Maybe use ispunct
        return is_uni_punctuation(word)
    else:
        return pos in punct_set or pos == 'PU'  # for chinese


def eval(batch_size, words, postags, heads_pred, types_pred, heads, types, lengths,
         punct_set=None, symbolic_root=False, symbolic_end=False):
    ucorr = 0.
    lcorr = 0.
    total = 0.
    ucomplete_match = 0.
    lcomplete_match = 0.

    ucorr_nopunc = 0.
    lcorr_nopunc = 0.
    total_nopunc = 0.
    ucomplete_match_nopunc = 0.
    lcomplete_match_nopunc = 0.

    corr_root = 0.
    total_root = 0.
    start = 1 if symbolic_root else 0
    end = 1 if symbolic_end else 0
    for i in range(batch_size):
        ucm = 1.
        lcm = 1.
        ucm_nopunc = 1.
        lcm_nopunc = 1.
        # assert len(heads[i]) == len(heads_pred[i])
        for j in range(start, lengths[i] - end):
            word = words[i][j]

            pos = postags[i][j]

            total += 1
            if heads[i][j] == heads_pred[i][j]:
                ucorr += 1
                if types[i][j] == types_pred[i][j]:
                    lcorr += 1
                else:
                    lcm = 0
            else:
                ucm = 0
                lcm = 0

            if not is_punctuation(word, pos, punct_set):
                total_nopunc += 1
                if heads[i][j] == heads_pred[i][j]:
                    ucorr_nopunc += 1
                    if types[i][j] == types_pred[i][j]:
                        lcorr_nopunc += 1
                    else:
                        lcm_nopunc = 0
                else:
                    ucm_nopunc = 0
                    lcm_nopunc = 0

            if heads_pred[i][j] == 0:
                total_root += 1
                corr_root += 1 if int(heads[i][j]) == 0 else 0

        ucomplete_match += ucm
        lcomplete_match += lcm
        ucomplete_match_nopunc += ucm_nopunc
        lcomplete_match_nopunc += lcm_nopunc

    return (ucorr, lcorr, total, ucomplete_match, lcomplete_match), \
           (ucorr_nopunc, lcorr_nopunc, total_nopunc, ucomplete_match_nopunc, lcomplete_match_nopunc), \
           (corr_root, total_root), batch_size


class SimpleAttachmentScore(Metric):

    def __init__(self, uas, las) -> None:
        super().__init__()
        self.las = las
        self.uas = uas

    @property
    def score(self):
        return self.las

    def __call__(self, pred, gold):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def __repr__(self):
        return f"UAS: {self.uas:.2%} LAS: {self.las:.2%}"
