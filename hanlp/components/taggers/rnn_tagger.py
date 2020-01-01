# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-09-14 20:30
from typing import Union, List

import tensorflow as tf

from hanlp.common.transform import Transform
from hanlp.components.taggers.tagger import TaggerComponent
from hanlp.transform.tsv import TSVTaggingTransform
from hanlp.common.vocab import Vocab
from hanlp.layers.embeddings import build_embedding, embeddings_require_string_input, embeddings_require_char_input
from hanlp.utils.util import merge_locals_kwargs


class RNNTagger(TaggerComponent):

    def __init__(self, transform: Transform = None) -> None:
        if not transform:
            self.transform = transform = TSVTaggingTransform()
        super().__init__(transform)

    def fit(self, trn_data: str, dev_data: str = None, save_dir: str = None, embeddings=100, embedding_trainable=False,
            rnn_input_dropout=0.2, rnn_units=100, rnn_output_dropout=0.2, epochs=20, lower=False, logger=None,
            loss: Union[tf.keras.losses.Loss, str] = None,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam', metrics='accuracy',
            batch_size=32, dev_batch_size=32, lr_decay_per_epoch=None, verbose=True, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_model(self, embeddings, embedding_trainable, rnn_input_dropout, rnn_output_dropout, rnn_units,
                    loss,
                    **kwargs) -> tf.keras.Model:
        model = tf.keras.Sequential()
        embeddings = build_embedding(embeddings, self.transform.word_vocab, self.transform)
        model.add(embeddings)
        if rnn_input_dropout:
            model.add(tf.keras.layers.Dropout(rnn_input_dropout, name='rnn_input_dropout'))
        model.add(
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=rnn_units, return_sequences=True), name='bilstm'))
        if rnn_output_dropout:
            model.add(tf.keras.layers.Dropout(rnn_output_dropout, name='rnn_output_dropout'))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(self.transform.tag_vocab)), name='dense'))
        return model

    def predict(self, sents: Union[List[str], List[List[str]]], batch_size=32, **kwargs) -> Union[
        List[str], List[List[str]]]:
        return super().predict(sents, batch_size)

    def save_weights(self, save_dir, filename='model.h5'):
        # remove the pre-trained embedding
        embedding_layer: tf.keras.layers.Embedding = self.model.get_layer(index=0)
        if embedding_layer.trainable:
            super().save_weights(save_dir, filename)
        else:
            truncated_model = tf.keras.Sequential(layers=self.model.layers[1:])
            truncated_model.build(input_shape=embedding_layer.output_shape)
            truncated_model.save_weights(save_dir)

    def build_loss(self, loss, **kwargs):
        if not loss:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM,
                                                                 from_logits=True)
            return loss
        return super().build_loss(loss, **kwargs)

    @property
    def tag_vocab(self) -> Vocab:
        return self.transform.tag_vocab

    def build_transform(self, embeddings, **kwargs):
        if embeddings_require_string_input(embeddings):
            self.transform.map_x = False
            if embeddings_require_char_input(embeddings):
                self.transform.char_vocab = Vocab()
        return super().build_transform(**kwargs)

    @property
    def sample_data(self):
        if self.transform.char_vocab:
            # You cannot build your model by calling `build` if your layers do not support float type inputs.
            # Instead, in order to instantiate and build your model, `call` your model on real tensor data (of the
            # correct dtype).
            sample = tf.constant([
                ['hello', 'world', self.transform.word_vocab.pad_token],
                ['hello', 'this', 'world'],
            ])
            sample._keras_mask = tf.not_equal(sample, self.transform.word_vocab.pad_token)
            return sample
