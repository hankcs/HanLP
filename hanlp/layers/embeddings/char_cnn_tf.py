# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-20 21:15
from functools import reduce

import tensorflow as tf

from hanlp.common.vocab_tf import VocabTF
from hanlp.utils.tf_util import hanlp_register


@hanlp_register
class CharCNNEmbeddingTF(tf.keras.layers.Layer):
    def __init__(self, word_vocab: VocabTF, char_vocab: VocabTF,
                 char_embedding=100,
                 kernel_size=3,
                 filters=50,
                 dropout=0.5,
                 trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.char_embedding = char_embedding
        self.filters = filters
        self.kernel_size = kernel_size
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.embedding = tf.keras.layers.Embedding(input_dim=len(self.char_vocab), output_dim=char_embedding,
                                                   trainable=True, mask_zero=True)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.cnn = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')

    def call(self, inputs: tf.Tensor, **kwargs):
        mask = tf.not_equal(inputs, self.word_vocab.pad_token)
        inputs = tf.ragged.boolean_mask(inputs, mask)
        chars = tf.strings.unicode_split(inputs, input_encoding='UTF-8')
        chars = chars.to_tensor(default_value=self.char_vocab.pad_token)
        chars = self.char_vocab.lookup(chars)
        embed = self.embedding(chars)
        weights = embed._keras_mask
        embed = self.dropout(embed)
        features = masked_conv1d_and_max(embed, weights, self.cnn)
        features._keras_mask = mask
        return features

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)

    def get_config(self):
        config = {
            'char_embedding': self.char_embedding,
            'kernel_size': self.kernel_size,
            'filters': self.filters,
            'dropout': self.dropout.rate,
        }
        base_config = super(CharCNNEmbeddingTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def masked_conv1d_and_max(t, weights, conv1d):
    """Applies 1d convolution and a masked max-pooling
    
    https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/masked_conv.py

    Args:
      t(tf.Tensor): A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
      weights(tf.Tensor of tf.bool): A Tensor of shape [d1, d2, dn-1]
      filters(int): number of filters
      kernel_size(int): kernel size for the temporal convolution
      conv1d: 

    Returns:

    
    """
    # Get shape and parameters
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x * y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.cast(weights, tf.float32)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = conv1d(t)
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
    t_max = tf.reduce_max(t_conv, axis=-2)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims - 2)] + [conv1d.filters]
    t_max = tf.reshape(t_max, shape=final_shape)

    return t_max
