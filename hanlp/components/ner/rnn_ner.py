# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-12 18:00
from typing import Any

import torch
from alnlp.metrics import span_utils

from hanlp.components.taggers.rnn_tagger import RNNTagger
from hanlp.metrics.chunking.conlleval import SpanF1
from hanlp_common.util import merge_locals_kwargs


class RNNNamedEntityRecognizer(RNNTagger):

    def __init__(self, **kwargs) -> None:
        """An old-school RNN tagger using word2vec or fasttext embeddings.

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)

    def build_metric(self, **kwargs):
        return SpanF1(self.tagging_scheme)

    def evaluate_dataloader(self, data, criterion, logger=None, ratio_width=None, **kwargs):
        loss, metric = super().evaluate_dataloader(data, criterion, logger, ratio_width, **kwargs)
        if logger:
            logger.info(metric.result(True, False)[-1])
        return loss, metric

    def fit(self, trn_data, dev_data, save_dir, batch_size=50, epochs=100, embed=100, rnn_input=None, rnn_hidden=256,
            drop=0.5, lr=0.001, patience=10, crf=True, optimizer='adam', token_key='token', tagging_scheme=None,
            anneal_factor: float = 0.5, delimiter=None, anneal_patience=2, devices=None,
            token_delimiter=None,
            logger=None,
            verbose=True, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def update_metrics(self, metric, logits, y, mask, batch, prediction):
        logits = self.decode_output(logits, mask, batch)
        if isinstance(logits, torch.Tensor):
            logits = logits.tolist()
        metric(self._id_to_tags(logits), batch['tag'])

    def predict(self, tokens: Any, batch_size: int = None, **kwargs):
        return super().predict(tokens, batch_size, **kwargs)

    def predict_data(self, data, batch_size, **kwargs):
        outputs = super().predict_data(data, batch_size)
        tagging_scheme = self.tagging_scheme
        if tagging_scheme == 'IOBES':
            entities = [span_utils.iobes_tags_to_spans(y) for y in outputs]
        elif tagging_scheme == 'BIO':
            entities = [span_utils.bio_tags_to_spans(y) for y in outputs]
        elif tagging_scheme == 'BIOUL':
            entities = [span_utils.bioul_tags_to_spans(y) for y in outputs]
        else:
            raise ValueError(f'Unrecognized tag scheme {tagging_scheme}')
        for i, (tokens, es) in enumerate(zip(data, entities)):
            outputs[i] = [(self.config.token_delimiter.join(tokens[b:e + 1]), t, b, e + 1) for t, (b, e) in es]
        return outputs

    def save_config(self, save_dir, filename='config.json'):
        if self.config.token_delimiter is None:
            self.config.token_delimiter = '' if all(
                [len(x) == 1 for x in self.vocabs[self.config.token_key].idx_to_token[-100:]]) else ' '
        super().save_config(save_dir, filename)
