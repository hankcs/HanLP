# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-26 23:05
# Ported from the PyTorch implementation https://github.com/zysite/biaffine-parser
import tensorflow as tf

from hanlp.utils.tf_util import tf_bernoulli


class Biaffine(tf.keras.layers.Layer):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = None

    def build(self, input_shape):
        self.weight = self.add_weight(name='kernel',
                                      shape=(self.n_out,
                                             self.n_in + self.bias_x,
                                             self.n_in + self.bias_y),
                                      initializer='zero')

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    # noinspection PyMethodOverriding
    def call(self, x, y, **kwargs):
        if self.bias_x:
            x = tf.concat((x, tf.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = tf.concat((y, tf.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = tf.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        if self.n_out == 1:
            s = tf.squeeze(s, axis=1)

        return s


class MLP(tf.keras.layers.Layer):
    def __init__(self, n_hidden, dropout=0, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.linear = tf.keras.layers.Dense(n_hidden, kernel_initializer='orthogonal')
        self.activation = tf.keras.layers.LeakyReLU(0.1)
        self.dropout = SharedDropout(p=dropout)

    def call(self, x, **kwargs):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class SharedDropout(tf.keras.layers.Layer):

    def __init__(self, p=0.5, batch_first=True, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def call(self, x, training=None, **kwargs):
        if training and self.p > 0:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= tf.expand_dims(mask, axis=1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = tf_bernoulli(tf.shape(x), 1 - p)
        mask = mask / (1 - p)

        return mask


class IndependentDropout(tf.keras.layers.Layer):

    def __init__(self, p=0.5, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def call(self, inputs, training=None, **kwargs):
        if training and self.p > 0:
            masks = [tf_bernoulli(tf.shape(x)[:2], 1 - self.p)
                     for x in inputs]
            total = sum(masks)
            scale = len(inputs) / tf.reduce_max(tf.ones_like(total))
            masks = [mask * scale for mask in masks]
            inputs = [item * tf.expand_dims(mask, axis=-1)
                      for item, mask in zip(inputs, masks)]

        return inputs
