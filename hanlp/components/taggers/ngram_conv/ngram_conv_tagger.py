# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-25 00:04

from typing import Union, Optional, Tuple, Any, Iterable, List

import tensorflow as tf

from hanlp.common.structure import SerializableDict
from hanlp.components.taggers.tagger import TaggerComponent
from hanlp.transform.tsv import TSVTaggingTransform
from hanlp.transform.txt import extract_ngram_features, bmes_to_words
from hanlp.common.vocab import Vocab
from hanlp.layers.embeddings import build_embedding
from hanlp.layers.weight_normalization import WeightNormalization
from hanlp.utils.util import merge_locals_kwargs


class NgramTransform(TSVTaggingTransform):

    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, **kwargs) -> None:
        super().__init__(config, map_x, map_y, **kwargs)
        self.ngram_vocab: Optional[Vocab] = None
        self.tag_vocab: Optional[Vocab] = None

    def inputs_to_samples(self, inputs, gold=False):
        for data in inputs:
            if gold:
                words, tags = data
            else:
                words, tags = data, [self.tag_vocab.safe_pad_token] * len(data)
            features = [words]
            if not tags:
                tags = [self.tag_vocab.first_token] * len(words)
            features.extend(extract_ngram_features(words, False, self.config.window_size))
            yield tuple(features), tags

    def x_to_idx(self, x) -> Union[tf.Tensor, Tuple]:
        ids = [self.word_vocab.lookup(x[0]) if self.config.map_word_feature else x[0]]
        for ngram in x[1:]:
            ids.append(self.ngram_vocab.lookup(ngram))
        return tuple(ids)

    def y_to_idx(self, y) -> tf.Tensor:
        return self.tag_vocab.lookup(y)

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        window_size = self.config.window_size
        ngram_size = window_size * (window_size + 1) // 2
        vec_dim = 2 + ngram_size
        shapes = tuple([[None]] * (vec_dim - 1)), [None]
        types = tuple([tf.string] * (vec_dim - 1)), tf.string
        word_vocab, ngram_vocab, tag_vocab = self.word_vocab, self.ngram_vocab, self.tag_vocab
        defaults = tuple([word_vocab.pad_token] + [
            ngram_vocab.pad_token if ngram_vocab else word_vocab.pad_token] * ngram_size), (
                       tag_vocab.pad_token if tag_vocab.pad_token else tag_vocab.first_token)
        return types, shapes, defaults

    def fit(self, trn_path: str, **kwargs):
        word_vocab, ngram_vocab, tag_vocab = Vocab(), Vocab(), Vocab(pad_token=None, unk_token=None)
        num_samples = 0
        for X, Y in self.file_to_samples(trn_path, gold=True):
            num_samples += 1
            word_vocab.update(X[0])
            for ngram in X[1:]:
                ngram_vocab.update(filter(lambda x: x, ngram))
            tag_vocab.update(Y)
        self.word_vocab, self.ngram_vocab, self.tag_vocab = word_vocab, ngram_vocab, tag_vocab
        if self.config.window_size:
            vocabs = word_vocab, ngram_vocab, tag_vocab
        else:
            vocabs = word_vocab, None, tag_vocab
        self.word_vocab, self.ngram_vocab, self.tag_vocab = vocabs
        return num_samples

    def X_to_inputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]]) -> Iterable:
        yield from super().X_to_inputs(X[0])

    def input_truth_output_to_str(self, input: List[str], truth: List[str], output: List[str]):
        words = bmes_to_words(input, output)
        return ' '.join(words)


class NgramConvTaggingModel(tf.keras.models.Model):
    def __init__(self, word_embed: tf.keras.layers.Embedding, ngram_embed: tf.keras.layers.Embedding, filters,
                 kernel_size, dropout_embed, dropout_hidden, weight_norm, num_tags, **kwargs):
        super().__init__(**kwargs)
        if ngram_embed is not None:
            self.ngram_embed = ngram_embed
        self.word_embed = word_embed
        # self.concat = tf.keras.layers.Concatenate(axis=2)
        self.dropout_embed = tf.keras.layers.Dropout(rate=dropout_embed)
        self.filters_w = []
        self.filters_v = []

        def create_conv1d(filter, name):
            conv = tf.keras.layers.Conv1D(filter, kernel_size, padding="same", name=name)
            if weight_norm:
                conv_norm = WeightNormalization(conv, name=name + '_norm', data_init=False)
                return conv_norm
            return conv

        for idx, filter in enumerate(filters):
            self.filters_w.append(create_conv1d(filter, 'Conv1Dw_{}'.format(idx)))
            self.filters_v.append(create_conv1d(filter, 'Conv1Dv_{}'.format(idx)))
        self.dropout_hidden = tf.keras.layers.Dropout(rate=dropout_hidden)
        self.dense = tf.keras.layers.Dense(num_tags, use_bias=False)

    def call(self, inputs, **kwargs):
        if hasattr(self, 'ngram_embed'):
            chars, ngrams = inputs[0], inputs[1:]
            embeds = [self.word_embed(chars)]
            mask = embeds[0]._keras_mask
            for ngram in ngrams:
                embeds.append(self.ngram_embed(ngram))
            if len(embeds) > 1:
                embed_input = tf.concat(embeds, axis=2)
            else:
                embed_input = embeds[0]
        else:
            chars = inputs if isinstance(inputs, tf.Tensor) else inputs[0]
            embed_input = self.word_embed(chars)
            mask = embed_input._keras_mask

        mask_float = tf.dtypes.cast(mask, tf.float32)
        embed_input = self.dropout_embed(embed_input)
        hidden_output = embed_input
        for fw, fv in zip(self.filters_w.layers, self.filters_v.layers):
            w = fw(hidden_output)
            v = fv(hidden_output)
            hidden_output = w * tf.nn.sigmoid(v)
            # Mask paddings.
            hidden_output = hidden_output * tf.expand_dims(mask_float, -1)
            hidden_output = self.dropout_hidden(hidden_output)
        # dirty hack
        hidden_output._keras_mask = mask
        logits = self.dense(hidden_output)
        return logits


class NgramConvTagger(TaggerComponent):

    def __init__(self, transform: NgramTransform = None) -> None:
        if not transform:
            transform = NgramTransform()
        super().__init__(transform)
        self.transform: NgramTransform = transform

    def build_model(self, word_embed, ngram_embed, window_size, weight_norm, filters, kernel_size, dropout_embed,
                    dropout_hidden, **kwargs) -> tf.keras.Model:
        word_vocab, ngram_vocab, tag_vocab = self.transform.word_vocab, self.transform.ngram_vocab, \
                                             self.transform.tag_vocab
        word_embed = build_embedding(word_embed, word_vocab, self.transform)
        if 'map_x' in self.config:
            self.config.map_word_feature = self.config.map_x
            del self.config.map_x
        else:
            self.config.map_word_feature = True
        if window_size:
            ngram_embed = build_embedding(ngram_embed, ngram_vocab, self.transform)
        else:
            ngram_embed = None
        model = NgramConvTaggingModel(word_embed, ngram_embed, filters, kernel_size, dropout_embed, dropout_hidden,
                                      weight_norm, len(tag_vocab))

        return model

    def fit(self, trn_data: Any, dev_data: Any, save_dir: str, word_embed: Union[str, int, dict] = 200,
            ngram_embed: Union[str, int,dict] = 50, embedding_trainable=True, window_size=4, kernel_size=3,
            filters=(200, 200, 200, 200, 200), dropout_embed=0.2, dropout_hidden=0.2, weight_norm=True,
            loss: Union[tf.keras.losses.Loss, str] = None,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam', metrics='accuracy', batch_size=100,
            epochs=100,
            logger=None, verbose=True, **kwargs):
        assert kwargs.get('run_eagerly', True), 'NgramConvTaggingModel can only run eagerly'
        kwargs['run_eagerly'] = True
        return super().fit(**merge_locals_kwargs(locals(), kwargs))
