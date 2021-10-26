# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-11 22:14
import io
from collections import defaultdict
from typing import List, Set, Tuple, Dict

from hanlp.metrics.chunking.conlleval import calculate_metrics, DetailedF1, metrics
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


class DetailedSpanF1(Metric):
    def __init__(self, do_confusion_matrix=False):
        self.correct_chunk = 0  # number of correctly identified chunks
        self.correct_unlabeled = 0
        self.total_gold = 0  # number of chunks in corpus
        self.total_pred = 0  # number of identified chunks
        self.token_counter = 0  # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_total_gold = defaultdict(int)
        self.t_total_pred = defaultdict(int)

        self.do_confusion_matrix = do_confusion_matrix
        if do_confusion_matrix:
            self.pred_labels = []
            self.gold_labels = []

    @property
    def states(self):
        return (self.t_correct_chunk, self.t_total_gold, self.t_total_pred)

    def reset_state(self):
        self.correct_chunk = 0  # number of correctly identified chunks
        self.total_gold = 0  # number of chunks in corpus
        self.total_pred = 0  # number of identified chunks
        self.token_counter = 0  # token counter (ignores sentence breaks)
        for state in self.states:
            state.clear()
        if self.do_confusion_matrix:
            self.pred_labels = []
            self.gold_labels = []

    @property
    def score(self):
        overall = calculate_metrics(
            self.correct_chunk, self.total_pred, self.total_gold
        )
        return overall.fscore

    def __call__(self, pred: Set[Tuple[int, int, str]], gold: Set[Tuple[int, int, str]], num_tokens=None):
        pred_chunks_unlabeled = set((b, e) for b, e, l in pred)
        gold_chunks_unlabeled = set((b, e) for b, e, l in gold)
        self.correct_unlabeled += len(pred_chunks_unlabeled & gold_chunks_unlabeled)
        self.correct_chunk += len(pred & gold)
        self.total_gold += len(gold)
        self.total_pred += len(pred)
        if num_tokens:
            self.token_counter += num_tokens

        def group_by_tag(collection: Set[Tuple[int, int, str]]):
            group = defaultdict(set)
            for b, e, l in collection:
                group[l].add((b, e))
            return group

        pred_tags = group_by_tag(pred)
        gold_tags = group_by_tag(gold)
        for l in pred_tags.keys() | gold_tags.keys():
            self.t_correct_chunk[l] += len(pred_tags[l] & gold_tags[l])
            self.t_total_gold[l] += len(gold_tags[l])
            self.t_total_pred[l] += len(pred_tags[l])

        if self.do_confusion_matrix:
            def group_by_span(collection: Set[Tuple[int, int, str]]):
                group = dict()
                for b, e, l in collection:
                    group[(b, e)] = l
                return group

            pred_spans = group_by_span(pred)
            gold_spans = group_by_span(gold)
            for span in pred_spans.keys() & gold_spans.keys():
                self.pred_labels.append(pred_spans[span])
                self.gold_labels.append(gold_spans[span])

    def reset(self):
        self.reset_state()

    def report(self) -> Tuple[DetailedF1, Dict[str, DetailedF1], str]:
        out = io.StringIO()

        c = self
        out.write('processed %d tokens with %d phrases; ' % (c.token_counter, c.total_gold))
        out.write('found: %d phrases; correct: %d.\n' % (c.total_pred, c.correct_chunk))

        overall = calculate_metrics(c.correct_unlabeled, c.total_pred, c.total_gold)
        out.write('%17s: ' % 'unlabeled overall')
        out.write('precision: %6.2f%%; ' % (100. * overall.prec))
        out.write('recall: %6.2f%%; ' % (100. * overall.rec))
        out.write('FB1: %6.2f\n' % (100. * overall.fscore))

        overall, by_type = metrics(self)
        out.write('%17s: ' % 'labeled overall')
        out.write('precision: %6.2f%%; ' % (100. * overall.prec))
        out.write('recall: %6.2f%%; ' % (100. * overall.rec))
        out.write('FB1: %6.2f\n' % (100. * overall.fscore))

        for i, m in sorted(by_type.items()):
            out.write('%17s: ' % i)
            out.write('precision: %6.2f%%; ' % (100. * m.prec))
            out.write('recall: %6.2f%%; ' % (100. * m.rec))
            out.write('FB1: %6.2f  %d\n' % (100. * m.fscore, c.t_total_pred[i]))
        text = out.getvalue()
        out.close()
        return overall, by_type, text

    def __str__(self) -> str:
        return self.report()[-1]

    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        labels = sorted(self.gold_labels + self.pred_labels)
        return confusion_matrix(self.gold_labels, self.pred_labels, labels=labels), labels
