# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-31 13:52
from abc import ABC
from typing import Union, Tuple, Any, List, Iterable

import tensorflow as tf

from hanlp.components.taggers.tagger_tf import TaggerComponent
from hanlp.transform.tsv import TSVTaggingTransform
from hanlp.common.vocab_tf import VocabTF
from hanlp.layers.embeddings.util_tf import build_embedding


class WindowTokenTransform(TSVTaggingTransform):

    def fit(self, trn_path: str, **kwargs):
        self.word_vocab = VocabTF()
        self.tag_vocab = VocabTF(pad_token=None, unk_token=None)
        for ngrams, tags in self.file_to_samples(trn_path):
            for words in ngrams:
                self.word_vocab.update(words)
            self.tag_vocab.update(tags)

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        window_radius = self.config.window_radius
        window_size = 2 * window_radius + 1
        types = tf.string, tf.string
        shapes = [None, window_size], [None]
        values = self.word_vocab.pad_token, self.tag_vocab.first_token
        return types, shapes, values

    def inputs_to_samples(self, inputs, gold=False):
        window_radius = self.config.window_radius
        for t in inputs:
            if gold:
                words, tags = t
            else:
                words, tags = t, [self.padding_values[-1]] * len(t)
            ngrams = []
            for i, word in enumerate(words):
                features = []
                for t in range(-window_radius, window_radius + 1):
                    index = i + t
                    if index < 0:
                        feature = 'bos{}'.format(index)
                    elif index >= len(words):
                        feature = 'eos+{}'.format(index - len(words) + 1)
                    else:
                        feature = words[index]
                    features.append(feature)
                ngrams.append(features)
            yield ngrams, tags

    def X_to_inputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]]) -> Iterable:
        for xs in X:
            words = []
            for x in xs:
                words.append(self.word_vocab.idx_to_token[int(x[len(x) // 2])])
            yield words


class CNNTaggingModel(tf.keras.models.Model):
    def __init__(self, filters, num_tags, embed, dropout, kernels, **kwargs):
        super().__init__()
        self.embed = embed
        self.embed_dropout = tf.keras.layers.Dropout(rate=dropout)
        self.conv2d = []
        for k in kernels:
            self.conv2d.append(
                tf.keras.layers.Conv2D(filters=filters, kernel_size=k, data_format='channels_last', padding='same'))
        self.conv2d_dropout = tf.keras.layers.Dropout(rate=dropout)
        self.concat = tf.keras.layers.Concatenate()
        self.dense = tf.keras.layers.Dense(units=num_tags)

    def call(self, inputs, **kwargs):
        # if inputs.shape_h[0] is None:
        #     return tf.zeros_like()
        #     print(inputs)
        embeds = self.embed(inputs)
        embeds = self.embed_dropout(embeds)
        hs = [conv(embeds) for conv in self.conv2d]
        h = self.concat(hs)
        h = self.conv2d_dropout(h)
        shape_h = tf.shape(h)
        h = tf.reshape(h, [shape_h[0], shape_h[1], h.shape[2] * h.shape[3]])
        o = self.dense(h)
        if h.shape[0]:
            mask = embeds._keras_mask[:, :, 0]
            o._keras_mask = mask
        return o


class CNNTaggerTF(TaggerComponent, ABC):
    def __init__(self, transform: WindowTokenTransform = None) -> None:
        if not transform:
            transform = WindowTokenTransform()
        super().__init__(transform)
        self.model: CNNTaggingModel = self.model  # refine the type
        self.transform: WindowTokenTransform = self.transform

    def build_model(self, embedding, **kwargs) -> tf.keras.Model:
        embed = build_embedding(embedding, self.transform.word_vocab, self.transform)
        self.transform.map_x = embed.dtype != tf.string
        model = CNNTaggingModel(num_tags=len(self.transform.tag_vocab),
                                embed=embed,
                                **kwargs)
        # model.build((None, None, 3))
        return model

    # noinspection PyMethodOverriding
    def fit(self, trn_data: Any, dev_data: Any, save_dir: str, embedding=200, window_radius=3,
            kernels=(1, 2, 3, 4, 5), filters=200, dropout=0.3,
            loss: Union[tf.keras.losses.Loss, str] = None,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam', metrics='accuracy', batch_size=100,
            epochs=100,
            logger=None, verbose=True, **kwargs):
        kwargs.update(locals())
        for k in 'self', 'kwargs', '__class__':
            kwargs.pop(k)
        super().fit(**kwargs)

    @property
    def input_shape(self) -> List:
        return [[None, None, self.config.window_radius * 2 + 1]]
