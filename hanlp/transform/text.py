# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-04 11:46
from typing import Union, Tuple, Iterable, Any

import tensorflow as tf

from hanlp_common.structure import SerializableDict
from hanlp.common.transform_tf import Transform
from hanlp.common.vocab_tf import VocabTF
from hanlp.metrics.chunking.sequence_labeling import get_entities
from hanlp.utils.file_read_backwards import FileReadBackwards
from hanlp.utils.io_util import read_tsv_as_sents


class TextTransform(Transform):

    def __init__(self,
                 forward=True,
                 seq_len=10,
                 tokenizer='char',
                 config: SerializableDict = None, map_x=True, map_y=True, **kwargs) -> None:
        super().__init__(config, map_x, map_y, seq_len=seq_len, tokenizer=tokenizer, forward=forward, **kwargs)
        self.vocab: VocabTF = None

    def tokenize_func(self):
        if self.config.tokenizer == 'char':
            return list
        elif self.config.tokenizer == 'whitespace':
            return lambda x: x.split()
        else:
            return lambda x: x.split(self.config.tokenizer)

    def fit(self, trn_path: str, **kwargs) -> int:
        self.vocab = VocabTF()
        num_samples = 0
        for x, y in self.file_to_inputs(trn_path):
            self.vocab.update(x)
            num_samples += 1
        return num_samples

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        types = tf.string, tf.string
        shapes = [None], [None]
        defaults = self.vocab.pad_token, self.vocab.pad_token
        return types, shapes, defaults

    def file_to_inputs(self, filepath: str, gold=True):
        forward = self.config.forward
        seq_len = self.config.seq_len
        buffer = []
        tokenizer = self.tokenize_func()
        with open(filepath, encoding='utf-8') if forward else FileReadBackwards(filepath, encoding="utf-8") as src:
            for line in src:
                tokens = tokenizer(line)
                buffer += tokens
                while len(buffer) > seq_len:
                    yield buffer[:seq_len], buffer[1:1 + seq_len]
                    buffer.pop(0)

    def inputs_to_samples(self, inputs, gold=False):
        forward = self.config.forward
        for t in inputs:
            if gold:
                x, y = t
            else:
                x, y = t, t
            if not forward:
                x = list(reversed(x))
                y = list(reversed(y))
            yield x, y

    def x_to_idx(self, x) -> Union[tf.Tensor, Tuple]:
        return self.vocab.lookup(x)

    def y_to_idx(self, y) -> tf.Tensor:
        return self.x_to_idx(y)

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, **kwargs) -> Iterable:
        pred = tf.argmax(Y, axis=-1)
        for ys, ms in zip(pred, inputs):
            ret = []
            for y in ys:
                ret.append(self.vocab.idx_to_token[int(y)])
            yield ret

    def input_is_single_sample(self, input: Any) -> bool:
        return isinstance(input[0], str)


def bmes_to_flat(inpath, outpath):
    with open(outpath, 'w', encoding='utf-8') as out:
        for sent in read_tsv_as_sents(inpath):
            chunks = get_entities([cells[1] for cells in sent])
            chars = [cells[0] for cells in sent]
            words = []
            for tag, start, end in chunks:
                word = ''.join(chars[start: end])
                words.append(word)
            out.write(' '.join(f'{word}/{tag}' for word, (tag, _, _) in zip(words, chunks)))
            out.write('\n')