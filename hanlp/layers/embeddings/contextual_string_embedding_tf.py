# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-19 03:24
from typing import List

import tensorflow as tf
import numpy as np
from hanlp.components.rnn_language_model import RNNLanguageModel
from hanlp_common.constant import PAD
from hanlp.utils.io_util import get_resource
from hanlp.utils.tf_util import copy_mask, hanlp_register, str_tensor_2d_to_list
from hanlp_common.util import infer_space_after


@hanlp_register
class ContextualStringEmbeddingTF(tf.keras.layers.Layer):

    def __init__(self, forward_model_path=None, backward_model_path=None, max_word_len=10,
                 trainable=False, name=None, dtype=None,
                 dynamic=True, **kwargs):
        assert dynamic, 'ContextualStringEmbedding works only in eager mode'
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        assert any([forward_model_path, backward_model_path]), 'At least one model is required'
        self.forward_model_path = forward_model_path
        self.backward_model_path = backward_model_path
        self.forward_model = self._load_lm(forward_model_path) if forward_model_path else None
        self.backward_model = self._load_lm(backward_model_path) if backward_model_path else None
        if trainable:
            self._fw = self.forward_model.model
            self._bw = self.backward_model.model
            for m in self._fw, self._bw:
                m.trainable = True
        self.supports_masking = True
        self.max_word_len = max_word_len

    def call(self, inputs, **kwargs):
        str_inputs = str_tensor_2d_to_list(inputs)
        outputs = self.embed(str_inputs)
        copy_mask(inputs, outputs)
        return outputs

    def _load_lm(self, filepath):
        filepath = get_resource(filepath)
        lm = RNNLanguageModel()
        lm.load(filepath)
        model: tf.keras.Sequential = lm.model
        for idx, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.LSTM):
                lm.model = tf.keras.Sequential(model.layers[:idx + 1])  # discard dense layer
                return lm

    def embed(self, texts: List[List[str]]):
        """Embedding sentences (list of words) with contextualized string embedding

        Args:
          texts: List of words, not chars
          texts: List[List[str]]: 

        Returns:

        
        """
        fw = None
        if self.forward_model:
            fw = self._run_rnn(texts, model=self.forward_model)
        bw = None
        if self.backward_model:
            bw = self._run_rnn(texts, model=self.backward_model)
        if not all(x is not None for x in [fw, bw]):
            return fw if fw is not None else bw
        else:
            return tf.concat([fw, bw], axis=-1)

    def _run_rnn(self, texts, model):
        embeddings = []
        inputs = []
        offsets = []
        tokenizer = model.transform.tokenize_func()
        backward = not model.config['forward']
        for sent in texts:
            raw, off = self._get_raw_string(sent, tokenizer)
            inputs.append(raw)
            offsets.append(off)
        outputs = model.model_from_config.predict(model.transform.inputs_to_dataset(inputs))
        if backward:
            outputs = tf.reverse(outputs, axis=[1])
        maxlen = len(max(texts, key=len))
        for hidden, off, sent in zip(outputs, offsets, texts):
            embed = []
            for (start, end), word in zip(off, sent):
                embed.append(hidden[end - 1, :])
            if len(embed) < maxlen:
                embed += [np.zeros_like(embed[-1])] * (maxlen - len(embed))
            embeddings.append(np.stack(embed))
        return tf.stack(embeddings)

    def _get_raw_string(self, sent: List[str], tokenizer):
        raw_string = []
        offsets = []
        whitespace_after = infer_space_after(sent)
        start = 0
        for word, space in zip(sent, whitespace_after):
            chars = tokenizer(word)
            chars = chars[:self.max_word_len]
            if space:
                chars += [' ']
            end = start + len(chars)
            offsets.append((start, end))
            start = end
            raw_string += chars
        return raw_string, offsets

    def get_config(self):
        config = {
            'forward_model_path': self.forward_model_path,
            'backward_model_path': self.backward_model_path,
            'max_word_len': self.max_word_len,
        }
        base_config = super(ContextualStringEmbeddingTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def output_dim(self):
        dim = 0
        for model in self.forward_model, self.backward_model:
            if model:
                dim += model.config['rnn_units']
        return dim

    def compute_output_shape(self, input_shape):
        return input_shape + self.output_dim

    def compute_mask(self, inputs, mask=None):

        return tf.not_equal(inputs, PAD)
