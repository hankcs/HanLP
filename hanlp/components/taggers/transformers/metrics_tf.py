# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-30 16:33
import tensorflow as tf


class Accuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, name='sparse_categorical_accuracy', dtype=None, mask_value=0):
        super().__init__(name, dtype)
        self.mask_value = mask_value

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_weight = tf.not_equal(y_true, self.mask_value)
        return super().update_state(y_true, y_pred, sample_weight)
