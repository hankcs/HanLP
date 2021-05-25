# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-20 16:25
from typing import Union, List, Callable

from hanlp.common.dataset import TransformableDataset
from hanlp.utils.io_util import read_cells

STS_B_TRAIN = 'http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz#sts-train.csv'
STS_B_DEV = 'http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz#sts-dev.csv'
STS_B_TEST = 'http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz#sts-test.csv'


class SemanticTextualSimilarityDataset(TransformableDataset):
    def __init__(self,
                 data: Union[str, List],
                 sent_a_col,
                 sent_b_col,
                 similarity_col,
                 delimiter='auto',
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None) -> None:
        self.delimiter = delimiter
        self.similarity_col = similarity_col
        self.sent_b_col = sent_b_col
        self.sent_a_col = sent_a_col
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath: str):
        for i, cells in enumerate(read_cells(filepath, strip=True, delimiter=self.delimiter)):
            yield {
                'sent_a': cells[self.sent_a_col],
                'sent_b': cells[self.sent_b_col],
                'similarity': float(cells[self.similarity_col])
            }
