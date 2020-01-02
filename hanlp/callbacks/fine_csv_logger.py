# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-05 02:12
import copy
from io import TextIOWrapper
from typing import List

import numpy as np
import tensorflow as tf


class StreamTableFormatter(object):

    def __init__(self) -> None:
        super().__init__()
        self.col_widths = None

    def format_row(self, cells) -> List[str]:
        if not isinstance(cells, list):
            cells = list(cells)
        if not self.col_widths:
            self.col_widths = [0] * len([_ for _ in cells])
        for i, c in enumerate(cells):
            self.col_widths[i] = max(self.col_widths[i], len(self.format_cell(c, self.col_widths[i])))
        return list(self.format_cell(cell, width) for cell, width in zip(cells, self.col_widths))

    def format_cell(self, cell: str, min_width) -> str:
        if isinstance(cell, (np.float32, np.float)):
            return '{:>{}.4f}'.format(cell, min_width)
        return '{:>{}}'.format(cell, min_width)


class FineCSVLogger(tf.keras.callbacks.History):

    def __init__(self, filename, separator=',', append=False):
        super().__init__()
        self.append = append
        self.separator = separator
        self.filename = filename
        self.out: TextIOWrapper = None
        self.keys = []
        self.formatter = StreamTableFormatter()

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.out = open(self.filename, 'a' if self.append else 'w')

    def on_train_end(self, logs=None):
        self.out.close()

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if not self.keys:
            self.keys = sorted(logs.keys())

            if getattr(self.model, 'stop_training', None):
                # We set NA so that csv parsers do not fail for this last epoch.
                logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

            # feed them twice to decide the actual width
            values = self.formatter.format_row([epoch + 1] + [logs.get(k, 'NA') for k in self.keys])
            headers = self.formatter.format_row(['epoch'] + self.keys)
            # print headers and bars
            self.out.write(self.separator.join(headers) + '\n')
            # bars for markdown style
            bars = [''.join(['-'] * width) for width in self.formatter.col_widths]
            self.out.write(self.separator.join(bars) + '\n')

        values = self.formatter.format_row([epoch + 1] + [logs.get(k, 'NA') for k in self.keys])
        self.out.write(self.separator.join(values) + '\n')
        self.out.flush()
