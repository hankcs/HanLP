# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-13 21:15
import functools
from abc import ABC
from typing import Tuple, Union, Optional, Iterable, List

import tensorflow as tf

from hanlp.common.structure import SerializableDict

from hanlp.common.transform import Transform
from hanlp.common.vocab import Vocab
from hanlp.utils.io_util import generator_words_tags
from hanlp.utils.tf_util import str_tensor_to_str
from hanlp.utils.util import merge_locals_kwargs


def dataset_from_tsv(tsv_file_path, word_vocab: Vocab, char_vocab: Vocab, tag_vocab: Vocab, batch_size=32,
                     shuffle=None, repeat=None, prefetch=1, lower=False, **kwargs):
    generator = functools.partial(generator_words_tags, tsv_file_path, word_vocab, char_vocab, tag_vocab, lower)
    return dataset_from_generator(generator, word_vocab, tag_vocab, batch_size, shuffle, repeat, prefetch,
                                  **kwargs)


def dataset_from_generator(generator, word_vocab, tag_vocab, batch_size=32, shuffle=None, repeat=None, prefetch=1,
                           **kwargs):
    shapes = [None], [None]
    types = tf.string, tf.string
    defaults = word_vocab.pad_token, tag_vocab.pad_token if tag_vocab.pad_token else tag_vocab.first_token
    dataset = tf.data.Dataset.from_generator(generator, output_shapes=shapes, output_types=types)
    if shuffle:
        if isinstance(shuffle, bool):
            shuffle = 1024
        dataset = dataset.shuffle(shuffle)
    if repeat:
        dataset = dataset.repeat(repeat)
    dataset = dataset.padded_batch(batch_size, shapes, defaults).prefetch(prefetch)
    return dataset


def vocab_from_tsv(tsv_file_path, lower=False, lock_word_vocab=False, lock_char_vocab=True, lock_tag_vocab=True) \
        -> Tuple[Vocab, Vocab, Vocab]:
    word_vocab = Vocab()
    char_vocab = Vocab()
    tag_vocab = Vocab(unk_token=None)
    with open(tsv_file_path, encoding='utf-8') as tsv_file:
        for line in tsv_file:
            cells = line.strip().split()
            if cells:
                word, tag = cells
                if lower:
                    word_vocab.add(word.lower())
                else:
                    word_vocab.add(word)
                char_vocab.update(list(word))
                tag_vocab.add(tag)
    if lock_word_vocab:
        word_vocab.lock()
    if lock_char_vocab:
        char_vocab.lock()
    if lock_tag_vocab:
        tag_vocab.lock()
    return word_vocab, char_vocab, tag_vocab


class TsvTaggingFormat(Transform, ABC):
    def file_to_inputs(self, filepath: str, gold=True):
        assert gold, 'TsvTaggingFormat does not support reading non-gold files'
        yield from generator_words_tags(filepath, gold=gold, lower=self.config.get('lower', False),
                                        max_seq_length=self.max_seq_length)

    @property
    def max_seq_length(self):
        return self.config.get('max_seq_length', None)


class TSVTaggingTransform(TsvTaggingFormat, Transform):
    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, use_char=False, **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.word_vocab: Optional[Vocab] = None
        self.tag_vocab: Optional[Vocab] = None
        self.char_vocab: Optional[Vocab] = None

    def fit(self, trn_path: str, **kwargs) -> int:
        self.word_vocab = Vocab()
        self.tag_vocab = Vocab(pad_token=None, unk_token=None)
        num_samples = 0
        for words, tags in self.file_to_inputs(trn_path, True):
            self.word_vocab.update(words)
            self.tag_vocab.update(tags)
            num_samples += 1
        if self.char_vocab:
            self.char_vocab = Vocab()
            for word in self.word_vocab.token_to_idx.keys():
                if word in (self.word_vocab.pad_token, self.word_vocab.unk_token):
                    continue
                self.char_vocab.update(list(word))
        return num_samples

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        types = tf.string, tf.string
        shapes = [None], [None]
        values = self.word_vocab.pad_token, self.tag_vocab.first_token
        return types, shapes, values

    def inputs_to_samples(self, inputs, gold=False):
        lower = self.config.get('lower', False)
        if gold:
            if lower:
                for x, y in inputs:
                    yield x.lower(), y
            else:
                yield from inputs
        else:
            for x in inputs:
                yield x.lower() if lower else x, [self.padding_values[-1]] * len(x)

    def x_to_idx(self, x) -> Union[tf.Tensor, Tuple]:
        return self.word_vocab.lookup(x)

    def y_to_idx(self, y) -> tf.Tensor:
        return self.tag_vocab.lookup(y)

    def X_to_inputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]]) -> Iterable:
        for xs in X:
            words = []
            for x in xs:
                words.append(str_tensor_to_str(x) if self.char_vocab else self.word_vocab.idx_to_token[int(x)])
            yield words

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False,
                     inputs=None, X=None, **kwargs) -> Iterable:
        if not gold:
            Y = tf.argmax(Y, axis=2)
        for ys, xs in zip(Y, inputs):
            tags = []
            for y, x in zip(ys, xs):
                tags.append(self.tag_vocab.idx_to_token[int(y)])
            yield tags

    def input_is_single_sample(self, input: Union[List[str], List[List[str]]]) -> bool:
        return isinstance(input[0], str)

    def input_truth_output_to_str(self, input: List[str], truth: List[str], output: List[str]):
        text = ''
        for word, gold_tag, pred_tag in zip(input, truth, output):
            text += ' '.join([word, gold_tag, pred_tag]) + '\n'

        text += '\n'
        return text
