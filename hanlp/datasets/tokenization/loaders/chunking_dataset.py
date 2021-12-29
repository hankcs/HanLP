# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-03 18:50
from typing import Union, List, Callable

from hanlp.common.dataset import TransformableDataset
from hanlp.utils.io_util import get_resource
from hanlp.utils.span_util import bmes_of
from hanlp.utils.string_util import ispunct


class ChunkingDataset(TransformableDataset):

    def __init__(self, data: Union[str, List], transform: Union[Callable, List] = None, cache=None,
                 generate_idx=None, max_seq_len=None, sent_delimiter=None) -> None:
        if not sent_delimiter:
            sent_delimiter = lambda x: ispunct(x)
        elif isinstance(sent_delimiter, str):
            sent_delimiter = set(list(sent_delimiter))
            sent_delimiter = lambda x: x in sent_delimiter
        self.sent_delimiter = sent_delimiter
        self.max_seq_len = max_seq_len
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath):
        max_seq_len = self.max_seq_len
        delimiter = self.sent_delimiter
        for chars, tags in self._generate_chars_tags(filepath, delimiter, max_seq_len):
            yield {'char': chars, 'tag': tags}

    @staticmethod
    def _generate_chars_tags(filepath, delimiter, max_seq_len):
        filepath = get_resource(filepath)
        with open(filepath, encoding='utf8') as src:
            for text in src:
                chars, tags = bmes_of(text, True)
                if max_seq_len and delimiter and len(chars) > max_seq_len:
                    short_chars, short_tags = [], []
                    for idx, (char, tag) in enumerate(zip(chars, tags)):
                        short_chars.append(char)
                        short_tags.append(tag)
                        if len(short_chars) >= max_seq_len and delimiter(char):
                            yield short_chars, short_tags
                            short_chars, short_tags = [], []
                    if short_chars:
                        yield short_chars, short_tags
                else:
                    yield chars, tags
