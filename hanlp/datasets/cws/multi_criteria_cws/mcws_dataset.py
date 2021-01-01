# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-10-21 19:11
import os
from typing import Union, List, Callable, Dict, Iterable

from hanlp.datasets.tokenization.txt import TextTokenizingDataset
from hanlp.utils.io_util import get_resource


class MultiCriteriaTextTokenizingDataset(TextTokenizingDataset):
    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None,
                 delimiter=None,
                 max_seq_len=None,
                 sent_delimiter=None,
                 char_level=False,
                 hard_constraint=False) -> None:
        super().__init__(data, transform, cache, generate_idx, delimiter, max_seq_len, sent_delimiter, char_level,
                         hard_constraint)

    def should_load_file(self, data) -> bool:
        return isinstance(data, (tuple, dict))

    def load_file(self, filepath: Union[Iterable[str], Dict[str, str]]):
        """Load multi-criteria corpora specified in filepath.

        Args:
            filepath: A list of files where filename is its criterion. Or a dict of filename-criterion pairs.

        .. highlight:: bash
        .. code-block:: bash

            $ tree -L 2 .
            .
            ├── cnc
            │   ├── dev.txt
            │   ├── test.txt
            │   ├── train-all.txt
            │   └── train.txt
            ├── ctb
            │   ├── dev.txt
            │   ├── test.txt
            │   ├── train-all.txt
            │   └── train.txt
            ├── sxu
            │   ├── dev.txt
            │   ├── test.txt
            │   ├── train-all.txt
            │   └── train.txt
            ├── udc
            │   ├── dev.txt
            │   ├── test.txt
            │   ├── train-all.txt
            │   └── train.txt
            ├── wtb
            │   ├── dev.txt
            │   ├── test.txt
            │   ├── train-all.txt
            │   └── train.txt
            └── zx
                ├── dev.txt
                ├── test.txt
                ├── train-all.txt
                └── train.txt

            $ head -n 2 ctb/dev.txt
            上海 浦东 开发 与 法制 建设 同步
            新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）

        """
        for eachpath in (filepath.items() if isinstance(filepath, dict) else filepath):
            if isinstance(eachpath, tuple):
                criteria, eachpath = eachpath
                eachpath = get_resource(eachpath)
            else:
                eachpath = get_resource(eachpath)
                criteria = os.path.basename(os.path.dirname(eachpath))
            for sample in super().load_file(eachpath):
                sample['criteria'] = criteria
                yield sample


def append_criteria_token(sample: dict, criteria_tokens: Dict[str, int], criteria_token_map: dict) -> dict:
    criteria = sample['criteria']
    token = criteria_token_map.get(criteria, None)
    if not token:
        unused_tokens = list(criteria_tokens.keys())
        size = len(criteria_token_map)
        assert size + 1 < len(unused_tokens), f'No unused token available for criteria {criteria}. ' \
                                              f'Current criteria_token_map = {criteria_token_map}'
        token = criteria_token_map[criteria] = unused_tokens[size]
    sample['token_token_type_ids'] = [0] * len(sample['token_input_ids']) + [1]
    sample['token_input_ids'] = sample['token_input_ids'] + [criteria_tokens[token]]
    return sample
