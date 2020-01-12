# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-09-14 20:33
from abc import ABC
from typing import Union, Any, Tuple, Iterable

import tensorflow as tf
from hanlp.components.taggers.transformers.transformer_transform import TransformerTransform

from hanlp.common.transform import Transform

from hanlp.common.component import KerasComponent
from hanlp.components.taggers.ngram_conv.ngram_conv_tagger import NgramConvTagger
from hanlp.components.taggers.rnn_tagger import RNNTagger
from hanlp.components.taggers.transformers.transformer_tagger import TransformerTagger
from hanlp.metrics.chunking.sequence_labeling import get_entities, iobes_to_span
from hanlp.utils.util import merge_locals_kwargs


class IOBES_NamedEntityRecognizer(KerasComponent, ABC):

    def predict_batch(self, batch, inputs=None):
        for words, tags in zip(inputs, super().predict_batch(batch, inputs)):
            yield from iobes_to_span(words, tags)


class IOBES_Transform(Transform):

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None) -> Iterable:
        for words, tags in zip(inputs, super().Y_to_outputs(Y, gold, inputs=inputs, X=X)):
            yield from iobes_to_span(words, tags)


class RNNNamedEntityRecognizer(RNNTagger, IOBES_NamedEntityRecognizer):

    def fit(self, trn_data: str, dev_data: str = None, save_dir: str = None, embeddings=100, embedding_trainable=False,
            rnn_input_dropout=0.2, rnn_units=100, rnn_output_dropout=0.2, epochs=20, logger=None,
            loss: Union[tf.keras.losses.Loss, str] = None,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam', metrics='f1', batch_size=32,
            dev_batch_size=32, lr_decay_per_epoch=None,
            run_eagerly=False,
            verbose=True, **kwargs):
        # assert kwargs.get('run_eagerly', True), 'This component can only run eagerly'
        # kwargs['run_eagerly'] = True
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_loss(self, loss, **kwargs):
        if not loss:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                from_logits=True)
        return super().build_loss(loss, **kwargs)


class NgramConvNamedEntityRecognizer(NgramConvTagger, IOBES_NamedEntityRecognizer):

    def fit(self, trn_data: Any, dev_data: Any, save_dir: str, word_embed: Union[str, int, dict] = 200,
            ngram_embed: Union[str, int, dict] = 50, embedding_trainable=True, window_size=4, kernel_size=3,
            filters=(200, 200, 200, 200, 200), dropout_embed=0.2, dropout_hidden=0.2, weight_norm=True,
            loss: Union[tf.keras.losses.Loss, str] = None,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam', metrics='f1', batch_size=100,
            epochs=100, logger=None, verbose=True, **kwargs):
        return super().fit(trn_data, dev_data, save_dir, word_embed, ngram_embed, embedding_trainable, window_size,
                           kernel_size, filters, dropout_embed, dropout_hidden, weight_norm, loss, optimizer, metrics,
                           batch_size, epochs, logger, verbose, **kwargs)


class IOBES_TransformerTransform(IOBES_Transform, TransformerTransform):
    pass


class TransformerNamedEntityRecognizer(TransformerTagger):

    def __init__(self, transform: TransformerTransform = None) -> None:
        if not transform:
            transform = IOBES_TransformerTransform()
        super().__init__(transform)

    def fit(self, trn_data, dev_data, save_dir, transformer, optimizer='adamw', learning_rate=5e-5, weight_decay_rate=0,
            epsilon=1e-8, clipnorm=1.0, warmup_steps_ratio=0, use_amp=False, max_seq_length=128, batch_size=32,
            epochs=3, metrics='f1', run_eagerly=False, logger=None, verbose=True, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))
