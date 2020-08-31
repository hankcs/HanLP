# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-20 01:29

import tensorflow as tf

from hanlp.utils.tf_util import hanlp_register


@hanlp_register
class SparseCategoricalCrossentropyOverNonzeroWeights(object):
    def __init__(self) -> None:
        super().__init__()
        self.__name__ = type(self).__name__

    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        if sample_weight is not None:
            loss = loss * sample_weight
        loss = tf.reduce_sum(loss)
        if sample_weight is not None:
            # This is equivalent to SUM_OVER_BATCH_SIZE
            # loss /= tf.reduce_sum(tf.ones_like(sample_weight, dtype=loss.dtype))
            # This one is SUM_BY_NONZERO_WEIGHTS
            loss /= tf.reduce_sum(sample_weight)
        return loss


@hanlp_register
class SparseCategoricalCrossentropyOverBatchFirstDim(object):

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = type(self).__name__

    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        if sample_weight is not None:
            loss = loss * sample_weight
        # could use sum of sample_weight[:,0] too
        loss = tf.reduce_sum(loss) / tf.cast(tf.shape(y_true)[0], tf.float32)
        return loss

    def get_config(self):
        return {}


@hanlp_register
class MaskedSparseCategoricalCrossentropyOverBatchFirstDim(object):
    def __init__(self, mask_value=0) -> None:
        super().__init__()
        self.mask_value = mask_value
        self.__name__ = type(self).__name__

    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        assert sample_weight is None, 'the mask will be computed via y_true != mask_value, ' \
                                      'it might conflict with sample_weight'
        active_loss = tf.not_equal(y_true, self.mask_value)
        active_labels = tf.boolean_mask(y_true, active_loss)
        active_logits = tf.boolean_mask(y_pred, active_loss)
        loss = tf.keras.losses.sparse_categorical_crossentropy(active_labels, active_logits, from_logits=True)
        loss = tf.reduce_sum(loss) / tf.cast(tf.shape(y_true)[0], tf.float32)
        return loss
