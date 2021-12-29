# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-04 17:28
from typing import List, Union

import tensorflow as tf

from hanlp.common.keras_component import KerasComponent
from hanlp.transform.text import TextTransform


class RNNLanguageModel(KerasComponent):

    def __init__(self, transform: TextTransform = None) -> None:
        if not transform:
            transform = TextTransform()
        super().__init__(transform)
        self.transform: TextTransform = transform

    def fit(self, trn_data, dev_data, save_dir,
            forward=True,
            embedding=100,
            rnn_input_dropout=0.1,
            rnn_units: int = 1024,
            rnn_output_dropout=0.1,
            seq_len: int = 250,
            optimizer='sgd',
            learning_rate=20,
            anneal_factor: float = 0.25,
            anneal_patience: int = 10,
            clipnorm=0.25,
            batch_size: int = 100, epochs=1000, run_eagerly=False, logger=None, verbose=True,
            **kwargs):
        return super().fit(**dict((k, v) for k, v in locals().items() if k not in ('self', 'kwargs')))

    def build_model(self, embedding, rnn_input_dropout, rnn_units, rnn_output_dropout, batch_size, seq_len, training,
                    **kwargs) -> tf.keras.Model:
        model = tf.keras.Sequential()
        extra_args = {}
        if training:
            extra_args['batch_input_shape'] = [batch_size, seq_len]
        embedding = tf.keras.layers.Embedding(input_dim=len(self.transform.vocab), output_dim=embedding,
                                              trainable=True, mask_zero=True, **extra_args)
        model.add(embedding)
        if rnn_input_dropout:
            model.add(tf.keras.layers.Dropout(rnn_input_dropout, name='rnn_input_dropout'))
        model.add(tf.keras.layers.LSTM(units=rnn_units, return_sequences=True, stateful=training, name='encoder'))
        if rnn_output_dropout:
            model.add(tf.keras.layers.Dropout(rnn_output_dropout, name='rnn_output_dropout'))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(self.transform.vocab)), name='decoder'))
        return model

    # noinspection PyMethodOverriding
    def build_optimizer(self, optimizer, learning_rate, clipnorm, **kwargs):
        if optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=clipnorm)
        return super().build_optimizer(optimizer, **kwargs)

    def build_train_dataset(self, trn_data, batch_size):
        trn_data = self.transform.file_to_dataset(trn_data, batch_size=batch_size, shuffle=False, repeat=-1)
        return trn_data

    def build_valid_dataset(self, dev_data, batch_size):
        dev_data = self.transform.file_to_dataset(dev_data, batch_size=batch_size, shuffle=False, drop_remainder=True)
        return dev_data

    def generate_text(self, text: Union[str, List[str]] = '\n', num_steps=50):
        char_mode = False
        if isinstance(text, str):
            text = list(text)
            char_mode = True
        forward = self.config['forward']
        # A slow implementation. Might better to let LSTM return states.
        # But anyway, this interface is for fun so let's take it easy
        for step in range(num_steps):
            output = self.predict(text)
            first_or_last_token = output[-1]
            if forward:
                text += first_or_last_token
            else:
                text = [first_or_last_token] + text
        if char_mode:
            text = ''.join(text)
        return text
