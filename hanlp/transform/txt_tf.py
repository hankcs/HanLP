# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-24 15:07
import functools
from abc import ABC
from typing import Tuple, Union, List, Iterable

import tensorflow as tf

from hanlp.common.transform_tf import Transform
from hanlp.common.vocab_tf import VocabTF
from hanlp.utils.io_util import get_resource
from hanlp.utils.lang.zh.char_table import CharTable
from hanlp.utils.span_util import bmes_of, bmes_to_words
from hanlp.utils.string_util import split_long_sent


def generate_words_per_line(file_path):
    with open(file_path, encoding='utf-8') as src:
        for line in src:
            cells = line.strip().split()
            if not cells:
                continue
            yield cells


def words_to_bmes(words):
    tags = []
    for w in words:
        if not w:
            raise ValueError('{} contains None or zero-length word {}'.format(str(words), w))
        if len(w) == 1:
            tags.append('S')
        else:
            tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
    return tags


def extract_ngram_features_and_tags(sentence, bigram_only=False, window_size=4, segmented=True):
    """
    Feature extraction for windowed approaches
    See Also https://github.com/chqiwang/convseg/
    Parameters
    ----------
    sentence
    bigram_only
    window_size
    segmented

    Returns
    -------

    """
    chars, tags = bmes_of(sentence, segmented)
    chars = CharTable.normalize_chars(chars)
    ret = []
    ret.append(chars)
    # TODO: optimize ngram generation using https://www.tensorflow.org/api_docs/python/tf/strings/ngrams
    ret.extend(extract_ngram_features(chars, bigram_only, window_size))
    ret.append(tags)
    return tuple(ret[:-1]), ret[-1]  # x, y


def extract_ngram_features(chars, bigram_only, window_size):
    ret = []
    if bigram_only:
        chars = ['', ''] + chars + ['', '']
        ret.append([a + b if a and b else '' for a, b in zip(chars[:-4], chars[1:])])
        ret.append([a + b if a and b else '' for a, b in zip(chars[1:-3], chars[2:])])
        ret.append([a + b if a and b else '' for a, b in zip(chars[2:-2], chars[3:])])
        ret.append([a + b if a and b else '' for a, b in zip(chars[3:-1], chars[4:])])
    elif window_size > 0:
        chars = ['', '', ''] + chars + ['', '', '']
        # single char
        if window_size >= 1:
            ret.append(chars[3:-3])
        if window_size >= 2:
            # bi chars
            ret.append([a + b if a and b else '' for a, b in zip(chars[2:], chars[3:-3])])
            ret.append([a + b if a and b else '' for a, b in zip(chars[3:-3], chars[4:])])
        if window_size >= 3:
            # tri chars
            ret.append(
                [a + b + c if a and b and c else '' for a, b, c in zip(chars[1:], chars[2:], chars[3:-3])])
            ret.append(
                [a + b + c if a and b and c else '' for a, b, c in zip(chars[2:], chars[3:-3], chars[4:])])
            ret.append(
                [a + b + c if a and b and c else '' for a, b, c in zip(chars[3:-3], chars[4:], chars[5:])])
        if window_size >= 4:
            # four chars
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                        zip(chars[0:], chars[1:], chars[2:], chars[3:-3])])
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                        zip(chars[1:], chars[2:], chars[3:-3], chars[4:])])
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                        zip(chars[2:], chars[3:-3], chars[4:], chars[5:])])
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                        zip(chars[3:-3], chars[4:], chars[5:], chars[6:])])
    return ret


def generate_ngram_bmes(file_path, bigram_only=False, window_size=4, gold=True):
    with open(file_path, encoding='utf-8') as src:
        for line in src:
            sentence = line.strip()
            if not sentence:
                continue
            yield extract_ngram_features_and_tags(sentence, bigram_only, window_size, gold)


def vocab_from_txt(txt_file_path, bigram_only=False, window_size=4, **kwargs) -> Tuple[VocabTF, VocabTF, VocabTF]:
    char_vocab, ngram_vocab, tag_vocab = VocabTF(), VocabTF(), VocabTF(pad_token=None, unk_token=None)
    for X, Y in generate_ngram_bmes(txt_file_path, bigram_only, window_size, gold=True):
        char_vocab.update(X[0])
        for ngram in X[1:]:
            ngram_vocab.update(filter(lambda x: x, ngram))
        tag_vocab.update(Y)
    return char_vocab, ngram_vocab, tag_vocab


def dataset_from_txt(txt_file_path: str, char_vocab: VocabTF, ngram_vocab: VocabTF, tag_vocab: VocabTF,
                     bigram_only=False,
                     window_size=4, segmented=True, batch_size=32, shuffle=None, repeat=None, prefetch=1):
    generator = functools.partial(generate_ngram_bmes, txt_file_path, bigram_only, window_size, segmented)
    return dataset_from_generator(generator, char_vocab, ngram_vocab, tag_vocab, bigram_only, window_size, batch_size,
                                  shuffle, repeat, prefetch)


def dataset_from_generator(generator, char_vocab, ngram_vocab, tag_vocab, bigram_only=False, window_size=4,
                           batch_size=32, shuffle=None, repeat=None, prefetch=1):
    if bigram_only:
        ngram_size = 4
    else:
        ngram_size = window_size * (window_size + 1) // 2
    vec_dim = 2 + ngram_size
    shapes = tuple([[None]] * (vec_dim - 1)), [None]
    types = tuple([tf.string] * (vec_dim - 1)), tf.string
    defaults = tuple([char_vocab.pad_token] + [
        ngram_vocab.pad_token if ngram_vocab else char_vocab.pad_token] * ngram_size), (
                   tag_vocab.pad_token if tag_vocab.pad_token else tag_vocab.first_token)
    dataset = tf.data.Dataset.from_generator(generator, output_shapes=shapes, output_types=types)
    if shuffle:
        if isinstance(shuffle, bool):
            shuffle = 1024
        dataset = dataset.shuffle(shuffle)
    if repeat:
        dataset = dataset.repeat(repeat)
    dataset = dataset.padded_batch(batch_size, shapes, defaults).prefetch(prefetch)
    return dataset


class TxtFormat(Transform, ABC):
    def file_to_inputs(self, filepath: str, gold=True):
        filepath = get_resource(filepath)
        with open(filepath, encoding='utf-8') as src:
            for line in src:
                sentence = line.strip()
                if not sentence:
                    continue
                yield sentence


class TxtBMESFormat(TxtFormat, ABC):
    def file_to_inputs(self, filepath: str, gold=True):
        max_seq_length = self.config.get('max_seq_length', False)
        if max_seq_length:
            if 'transformer' in self.config:
                max_seq_length -= 2  # allow for [CLS] and [SEP]
            delimiter = set()
            delimiter.update('。！？：；、，,;!?、,')
        for text in super().file_to_inputs(filepath, gold):
            chars, tags = bmes_of(text, gold)
            if max_seq_length:
                start = 0
                for short_chars in split_long_sent(chars, delimiter, max_seq_length):
                    end = start + len(short_chars)
                    yield short_chars, tags[start:end]
                    start = end
            else:
                yield chars, tags

    def input_is_single_sample(self, input: Union[List[str], List[List[str]]]) -> bool:
        return isinstance(input, str)

    def inputs_to_samples(self, inputs, gold=False):
        for chars, tags in (inputs if gold else zip(inputs, [None] * len(inputs))):
            if not gold:
                tags = [self.tag_vocab.safe_pad_token] * len(chars)
            chars = CharTable.normalize_chars(chars)
            yield chars, tags

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None,
                     batch=None) -> Iterable:
        yield from self.Y_to_tokens(self.tag_vocab, Y, gold, inputs)

    def Y_to_tokens(self, tag_vocab, Y, gold, inputs):
        if not gold:
            Y = tf.argmax(Y, axis=2)
        for text, ys in zip(inputs, Y):
            tags = [tag_vocab.idx_to_token[int(y)] for y in ys[:len(text)]]
            yield bmes_to_words(list(text), tags)
