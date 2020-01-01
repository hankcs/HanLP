# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-30 16:33
import tensorflow as tf
from tensorflow.python.keras.metrics import MeanMetricWrapper


def masked_sparse_categorical_accuracy(y_true, y_pred):
    mask = tf.not_equal(y_true, 0)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


class MaskedSparseCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, name='sparse_categorical_accuracy', dtype=None):
        super(MaskedSparseCategoricalAccuracy, self).__init__(
            masked_sparse_categorical_accuracy, name, dtype=dtype)
