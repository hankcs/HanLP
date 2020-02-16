# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-12 13:08
from typing import Any, Callable

from hanlp.components.taggers.rnn_tagger import RNNTagger
from hanlp.datasets.cws.chunking_dataset import ChunkingDataset
from hanlp.metrics.chunking.chunking_f1 import ChunkingF1
from hanlp.utils.span_util import bmes_to_words
from hanlp_common.util import merge_locals_kwargs


class RNNTokenizer(RNNTagger):

    def predict(self, sentence: Any, batch_size: int = None, **kwargs):
        flat = isinstance(sentence, str)
        if flat:
            sentence = [sentence]
        for i, s in enumerate(sentence):
            sentence[i] = list(s)
        outputs = RNNTagger.predict(self, sentence, batch_size, **kwargs)
        if flat:
            return outputs[0]
        return outputs

    def predict_data(self, data, batch_size, **kwargs):
        tags = RNNTagger.predict_data(self, data, batch_size, **kwargs)
        words = [bmes_to_words(c, t) for c, t in zip(data, tags)]
        return words

    def build_dataset(self, data, transform=None):
        dataset = ChunkingDataset(data)
        if 'transform' in self.config:
            dataset.append_transform(self.config.transform)
        if transform:
            dataset.append_transform(transform)
        return dataset

    def build_metric(self, **kwargs):
        return ChunkingF1()

    def update_metrics(self, metric, logits, y, mask, batch):
        pred = self.decode_output(logits, mask, batch)
        pred = self._id_to_tags(pred)
        gold = batch['tag']
        metric(pred, gold)

    def fit(self, trn_data, dev_data, save_dir, batch_size=50, epochs=100, embed=100, rnn_input=None, rnn_hidden=256,
            drop=0.5, lr=0.001, patience=10, crf=True, optimizer='adam', token_key='char', tagging_scheme=None,
            anneal_factor: float = 0.5, anneal_patience=2, devices=None, logger=None,
            verbose=True, transform: Callable = None, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))


