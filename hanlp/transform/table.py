# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-11-10 21:00
from abc import ABC
from typing import Tuple, Union

import tensorflow as tf

from hanlp.common.structure import SerializableDict
from hanlp.common.transform import Transform
from hanlp.common.constant import PAD
from hanlp.common.vocab import create_label_vocab
from hanlp.utils.io_util import read_cells
from hanlp.utils.log_util import logger


class TableTransform(Transform, ABC):
    def __init__(self, config: SerializableDict = None, map_x=False, map_y=True, x_columns=None,
                 y_column=-1,
                 skip_header=True, delimiter='auto', **kwargs) -> None:
        super().__init__(config, map_x, map_y, x_columns=x_columns, y_column=y_column,
                         skip_header=skip_header,
                         delimiter=delimiter, **kwargs)
        self.label_vocab = create_label_vocab()

    def file_to_inputs(self, filepath: str, gold=True):
        x_columns = self.config.x_columns
        y_column = self.config.y_column
        num_features = self.config.get('num_features', None)
        for cells in read_cells(filepath, skip_header=self.config.skip_header, delimiter=self.config.delimiter):
            if x_columns:
                inputs = tuple(c for i, c in enumerate(cells) if i in x_columns), cells[y_column]
            else:
                if y_column != -1:
                    cells[-1], cells[y_column] = cells[y_column], cells[-1]
                inputs = tuple(cells[:-1]), cells[-1]
            if num_features is None:
                num_features = len(inputs[0])
                self.config.num_features = num_features
            else:
                assert num_features == len(inputs[0]), f'Numbers of columns {num_features} ' \
                                                       f'inconsistent with current {len(inputs[0])}'
            yield inputs

    def inputs_to_samples(self, inputs, gold=False):
        pad = self.label_vocab.safe_pad_token
        for cells in inputs:
            if gold:
                yield cells
            else:
                yield cells, pad

    def y_to_idx(self, y) -> tf.Tensor:
        return self.label_vocab.lookup(y)

    def fit(self, trn_path: str, **kwargs):
        samples = 0
        for t in self.file_to_samples(trn_path, gold=True):
            self.label_vocab.add(t[1])  # the second one regardless of t is pair or triple
            samples += 1
        return samples

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        num_features = self.config.num_features
        # It's crucial to use tuple instead of list for all the three
        types = tuple([tf.string] * num_features), tf.string
        shapes = tuple([[]] * num_features), []
        values = tuple([PAD] * num_features), self.label_vocab.safe_pad_token
        return types, shapes, values

    def x_to_idx(self, x) -> Union[tf.Tensor, Tuple]:
        logger.warning('TableTransform can not map x to idx. Please override x_to_idx')
        return x
