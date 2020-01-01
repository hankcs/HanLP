# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-20 17:08
import tensorflow as tf

from hanlp.utils.tf_util import hanlp_register, copy_mask


@hanlp_register
class ConcatEmbedding(tf.keras.layers.Layer):
    def __init__(self, *embeddings, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        self.embeddings = []
        for embed in embeddings:
            embed: tf.keras.layers.Layer = tf.keras.utils.deserialize_keras_object(embed) if isinstance(embed,
                                                                                                        dict) else embed
            self.embeddings.append(embed)
            if embed.trainable:
                trainable = True
            if embed.dynamic:
                dynamic = True
            if embed.supports_masking:
                self.supports_masking = True

        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def build(self, input_shape):
        for embed in self.embeddings:
            embed.build(input_shape)
        super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        for embed in self.embeddings:
            mask = embed.compute_mask(inputs, mask)
            if mask is not None:
                return mask
        return mask

    def call(self, inputs, **kwargs):
        embeds = [embed.call(inputs) for embed in self.embeddings]
        feature = tf.concat(embeds, axis=-1)

        for embed in embeds:
            mask = copy_mask(embed, feature)
            if mask is not None:
                break
        return feature

    def get_config(self):
        config = {
            'embeddings': [embed.get_config() for embed in self.embeddings],
        }
        base_config = super(ConcatEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        dim = 0
        for embed in self.embeddings:
            dim += embed.compute_output_shape(input_shape)[-1]

        return input_shape + dim
