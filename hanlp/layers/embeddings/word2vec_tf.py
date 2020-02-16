# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-24 21:49
import os
from typing import Tuple, Union, List

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops

from hanlp.common.vocab_tf import VocabTF
from hanlp.utils.io_util import load_word2vec, get_resource
from hanlp.utils.tf_util import hanlp_register
from hanlp_common.util import DummyContext


class Word2VecEmbeddingV1(tf.keras.layers.Layer):
    def __init__(self, path: str = None, vocab: VocabTF = None, normalize: bool = False, load_all=True, mask_zero=True,
                 trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        if load_all and vocab and vocab.locked:
            vocab.unlock()
        self.vocab, self.array_np = self._load(path, vocab, normalize)
        self.vocab.lock()
        self.array_ks = tf.keras.layers.Embedding(input_dim=len(self.vocab), output_dim=self.dim, trainable=trainable,
                                                  embeddings_initializer=tf.keras.initializers.Constant(self.array_np),
                                                  mask_zero=mask_zero)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None

        return math_ops.not_equal(inputs, self.vocab.pad_idx)

    def call(self, inputs, **kwargs):
        return self.array_ks(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dim

    @staticmethod
    def _load(path, vocab, normalize=False) -> Tuple[VocabTF, Union[np.ndarray, None]]:
        if not vocab:
            vocab = VocabTF()
        if not path:
            return vocab, None
        assert vocab.unk_idx is not None

        word2vec, dim = load_word2vec(path)
        for word in word2vec:
            vocab.get_idx(word)

        pret_embs = np.zeros(shape=(len(vocab), dim), dtype=np.float32)
        state = np.random.get_state()
        np.random.seed(0)
        bias = np.random.uniform(low=-0.001, high=0.001, size=dim).astype(dtype=np.float32)
        scale = np.sqrt(3.0 / dim)
        for word, idx in vocab.token_to_idx.items():
            vec = word2vec.get(word, None)
            if vec is None:
                vec = word2vec.get(word.lower(), None)
                # if vec is not None:
                #     vec += bias
            if vec is None:
                # vec = np.random.uniform(-scale, scale, [dim])
                vec = np.zeros([dim], dtype=np.float32)
            pret_embs[idx] = vec
        # noinspection PyTypeChecker
        np.random.set_state(state)
        return vocab, pret_embs

    @property
    def size(self):
        if self.array_np is not None:
            return self.array_np.shape[0]

    @property
    def dim(self):
        if self.array_np is not None:
            return self.array_np.shape[1]

    @property
    def shape(self):
        if self.array_np is None:
            return None
        return self.array_np.shape

    def get_vector(self, word: str) -> np.ndarray:
        assert self.array_np is not None
        return self.array_np[self.vocab.get_idx_without_add(word)]

    def __getitem__(self, word: Union[str, List, tf.Tensor]) -> np.ndarray:
        if isinstance(word, str):
            return self.get_vector(word)
        elif isinstance(word, list):
            vectors = np.zeros(shape=(len(word), self.dim))
            for idx, token in enumerate(word):
                vectors[idx] = self.get_vector(token)
            return vectors
        elif isinstance(word, tf.Tensor):
            if word.dtype == tf.string:
                word_ids = self.vocab.token_to_idx_table.lookup(word)
                return tf.nn.embedding_lookup(self.array_tf, word_ids)
            elif word.dtype == tf.int32 or word.dtype == tf.int64:
                return tf.nn.embedding_lookup(self.array_tf, word)


@hanlp_register
class Word2VecEmbeddingTF(tf.keras.layers.Embedding):

    def __init__(self, filepath: str = None, vocab: VocabTF = None, expand_vocab=True, lowercase=True,
                 input_dim=None, output_dim=None, unk=None, normalize=False,
                 embeddings_initializer='VarianceScaling',
                 embeddings_regularizer=None,
                 activity_regularizer=None, embeddings_constraint=None, mask_zero=True, input_length=None,
                 name=None, cpu=True, **kwargs):
        filepath = get_resource(filepath)
        word2vec, _output_dim = load_word2vec(filepath)
        if output_dim:
            assert output_dim == _output_dim, f'output_dim = {output_dim} does not match {filepath}'
        output_dim = _output_dim
        # if the `unk` token exists in the pretrained,
        # then replace it with a self-defined one, usually the one in word vocab
        if unk and unk in word2vec:
            word2vec[vocab.safe_unk_token] = word2vec.pop(unk)
        if vocab is None:
            vocab = VocabTF()
            vocab.update(word2vec.keys())
        if expand_vocab and vocab.mutable:
            for word in word2vec:
                vocab.get_idx(word.lower() if lowercase else word)
        if input_dim:
            assert input_dim == len(vocab), f'input_dim = {input_dim} does not match {filepath}'
        input_dim = len(vocab)
        # init matrix
        self._embeddings_initializer = embeddings_initializer
        embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        with tf.device('cpu:0') if cpu else DummyContext():
            pret_embs = embeddings_initializer(shape=[input_dim, output_dim]).numpy()
        # insert to pret_embs
        for word, idx in vocab.token_to_idx.items():
            vec = word2vec.get(word, None)
            # Retry lower case
            if vec is None and lowercase:
                vec = word2vec.get(word.lower(), None)
            if vec is not None:
                pret_embs[idx] = vec
        if normalize:
            pret_embs /= np.std(pret_embs)
        if not name:
            name = os.path.splitext(os.path.basename(filepath))[0]
        super().__init__(input_dim, output_dim, tf.keras.initializers.Constant(pret_embs), embeddings_regularizer,
                         activity_regularizer, embeddings_constraint, mask_zero, input_length, name=name, **kwargs)
        self.filepath = filepath
        self.expand_vocab = expand_vocab
        self.lowercase = lowercase

    def get_config(self):
        config = {
            'filepath': self.filepath,
            'expand_vocab': self.expand_vocab,
            'lowercase': self.lowercase,
        }
        base_config = super(Word2VecEmbeddingTF, self).get_config()
        base_config['embeddings_initializer'] = self._embeddings_initializer
        return dict(list(base_config.items()) + list(config.items()))


@hanlp_register
class StringWord2VecEmbeddingTF(Word2VecEmbeddingTF):

    def __init__(self, filepath: str = None, vocab: VocabTF = None, expand_vocab=True, lowercase=False, input_dim=None,
                 output_dim=None, unk=None, normalize=False, embeddings_initializer='VarianceScaling',
                 embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=True,
                 input_length=None, name=None, **kwargs):
        if vocab is None:
            vocab = VocabTF()
        self.vocab = vocab
        super().__init__(filepath, vocab, expand_vocab, lowercase, input_dim, output_dim, unk, normalize,
                         embeddings_initializer, embeddings_regularizer, activity_regularizer, embeddings_constraint,
                         mask_zero, input_length, name, **kwargs)

    def call(self, inputs):
        assert inputs.dtype == tf.string, \
            f'Expect tf.string but got tf.{inputs.dtype.name}. {inputs}' \
            f'Please pass tf.{inputs.dtype.name} in.'
        inputs = self.vocab.lookup(inputs)
        # inputs._keras_mask = tf.not_equal(inputs, self.vocab.pad_idx)
        return super().call(inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, self.vocab.pad_token)
