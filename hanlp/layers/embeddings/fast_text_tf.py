# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-29 13:14
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils

from hanlp_common.constant import PAD
from hanlp.utils.io_util import get_resource, stdout_redirected
from hanlp.utils.log_util import logger
from hanlp.utils.tf_util import hanlp_register


@hanlp_register
class FastTextEmbeddingTF(tf.keras.layers.Embedding):

    def __init__(self, filepath: str, padding=PAD, name=None, **kwargs):
        import fasttext
        self.padding = padding.encode('utf-8')
        self.filepath = filepath
        filepath = get_resource(filepath)
        assert os.path.isfile(filepath), f'Resolved path {filepath} is not a file'
        logger.debug('Loading fasttext model from [{}].'.format(filepath))
        # fasttext print a blank line here
        with stdout_redirected(to=os.devnull, stdout=sys.stderr):
            self.model = fasttext.load_model(filepath)
        kwargs.pop('input_dim', None)
        kwargs.pop('output_dim', None)
        kwargs.pop('mask_zero', None)
        if not name:
            name = os.path.splitext(os.path.basename(filepath))[0]
        super().__init__(input_dim=len(self.model.words), output_dim=self.model['king'].size,
                         mask_zero=padding is not None, trainable=False, dtype=tf.string, name=name, **kwargs)
        embed_fn = np.frompyfunc(self.embed, 1, 1)
        # vf = np.vectorize(self.embed, otypes=[np.ndarray])
        self._embed_np = embed_fn

    def embed(self, word):
        return self.model[word]

    def embed_np(self, words: np.ndarray):
        output = self._embed_np(words)
        if self.mask_zero:
            mask = words != self.padding
            output *= mask
            output = np.stack(output.reshape(-1)).reshape(list(words.shape) + [self.output_dim])
            return output, tf.constant(mask)
        else:
            output = np.stack(output.reshape(-1)).reshape(list(words.shape) + [self.output_dim])
            return output

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.built = True

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

    def call(self, inputs: tf.Tensor):
        if isinstance(inputs, list):
            inputs = inputs[0]
        if not hasattr(inputs, 'numpy'):  # placeholder tensor
            inputs = tf.expand_dims(inputs, axis=-1)
            inputs = tf.tile(inputs, [1] * (len(inputs.shape) - 1) + [self.output_dim])
            inputs = tf.zeros_like(inputs, dtype=tf.float32)
            return inputs
            # seq_len = inputs.shape[-1]
            # if not seq_len:
            #     seq_len = 1
            # return tf.zeros([1, seq_len, self.output_dim])
        if self.mask_zero:
            outputs, masks = self.embed_np(inputs.numpy())
            outputs = tf.constant(outputs)
            outputs._keras_mask = masks
        else:
            outputs = self.embed_np(inputs.numpy())
            outputs = tf.constant(outputs)
        return outputs

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, self.padding)

    def get_config(self):
        config = {
            'filepath': self.filepath,
            'padding': self.padding.decode('utf-8')
        }
        base_config = super(FastTextEmbeddingTF, self).get_config()
        for junk in 'embeddings_initializer' \
                , 'batch_input_shape' \
                , 'embeddings_regularizer' \
                , 'embeddings_constraint' \
                , 'activity_regularizer' \
                , 'trainable' \
                , 'input_length' \
                :
            base_config.pop(junk)
        return dict(list(base_config.items()) + list(config.items()))
