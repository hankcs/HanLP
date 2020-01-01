# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-20 17:02
import tensorflow as tf

from hanlp.common.vocab import Vocab
from hanlp.utils.tf_util import hanlp_register


@hanlp_register
class CharRNNEmbedding(tf.keras.layers.Layer):
    def __init__(self, word_vocab: Vocab, char_vocab: Vocab,
                 char_embedding=100,
                 char_rnn_units=25,
                 dropout=0.5,
                 trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.char_embedding = char_embedding
        self.char_rnn_units = char_rnn_units
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.embedding = tf.keras.layers.Embedding(input_dim=len(self.char_vocab), output_dim=char_embedding,
                                                   trainable=True, mask_zero=True)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=char_rnn_units,
                                                                      return_state=True), name='bilstm')

    def call(self, inputs: tf.Tensor, **kwargs):
        mask = tf.not_equal(inputs, self.word_vocab.pad_token)
        inputs = tf.ragged.boolean_mask(inputs, mask)
        chars = tf.strings.unicode_split(inputs, input_encoding='UTF-8')
        chars = chars.to_tensor(default_value=self.char_vocab.pad_token)
        chars = self.char_vocab.lookup(chars)
        embed = self.embedding(chars)
        char_mask = embed._keras_mask
        embed = self.dropout(embed)
        embed_shape = tf.shape(embed)
        embed = tf.reshape(embed, [-1, embed_shape[2], embed_shape[3]])
        char_mask = tf.reshape(char_mask, [-1, embed_shape[2]])
        all_zeros = tf.reduce_sum(tf.cast(char_mask, tf.int32), axis=1) == 0
        char_mask_shape = tf.shape(char_mask)
        hole = tf.zeros(shape=(char_mask_shape[0], char_mask_shape[1] - 1), dtype=tf.bool)
        all_zeros = tf.expand_dims(all_zeros, -1)
        non_all_zeros = tf.concat([all_zeros, hole], axis=1)
        char_mask = tf.logical_or(char_mask, non_all_zeros)
        output, h_fw, c_fw, h_bw, c_bw = self.rnn(embed, mask=char_mask)
        hidden = tf.concat([h_fw, h_bw], axis=-1)
        # hidden = output
        hidden = tf.reshape(hidden, [embed_shape[0], embed_shape[1], -1])
        hidden._keras_mask = mask
        return hidden

    def get_config(self):
        config = {
            'char_embedding': self.char_embedding,
            'char_rnn_units': self.char_rnn_units,
            'dropout': self.dropout.rate,
        }
        base_config = super(CharRNNEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
