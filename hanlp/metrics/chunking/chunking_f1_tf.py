# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 23:09
from abc import ABC, abstractmethod

import tensorflow as tf

from hanlp.common.vocab_tf import VocabTF


class ChunkingF1_TF(tf.keras.metrics.Metric, ABC):

    def __init__(self, tag_vocab: VocabTF, from_logits=True, name='f1', dtype=None, **kwargs):
        super().__init__(name, dtype, dynamic=True, **kwargs)
        self.tag_vocab = tag_vocab
        self.from_logits = from_logits

    def update_the_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None, **kwargs):
        mask = y_pred._keras_mask if sample_weight is None else sample_weight
        if self.tag_vocab.pad_idx is not None and sample_weight is None:
            # in this case, the model doesn't compute mask but provide a masking index, it's ok to
            mask = y_true != self.tag_vocab.pad_idx
        assert mask is not None, 'ChunkingF1 requires masking, check your _keras_mask or compute_mask'
        if self.from_logits:
            y_pred = tf.argmax(y_pred, axis=-1)
        y_true = self.to_tags(y_true, mask)
        y_pred = self.to_tags(y_pred, mask)
        return self.update_tags(y_true, y_pred)

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None, **kwargs):
        return self.update_the_state(y_true, y_pred, sample_weight)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None, **kwargs):
        return self.update_the_state(y_true, y_pred, sample_weight)

    def to_tags(self, y: tf.Tensor, sample_weight: tf.Tensor):
        batch = []
        y = y.numpy()
        sample_weight = sample_weight.numpy()
        for sent, mask in zip(y, sample_weight):
            tags = []
            for tag, m in zip(sent, mask):
                if not m:
                    continue
                tag = int(tag)
                if self.tag_vocab.pad_idx is not None and tag == self.tag_vocab.pad_idx:
                    # If model predicts <pad>, it will fail most metrics. So replace it with a valid one
                    tag = 1
                tags.append(self.tag_vocab.get_token(tag))
            batch.append(tags)
        return batch

    @abstractmethod
    def update_tags(self, true_tags, pred_tags):
        pass

    @abstractmethod
    def result(self):
        pass
