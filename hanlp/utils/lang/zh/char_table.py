# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-09 19:07
from typing import List

from hanlp.utils.io_util import get_resource

HANLP_CHAR_TABLE = 'https://file.hankcs.com/corpus/char_table.zip#CharTable.txt'


class CharTable:
    convert = {}

    @staticmethod
    def convert_char(c):
        if not CharTable.convert:
            CharTable._init()
        return CharTable.convert.get(c, c)

    @staticmethod
    def normalize_text(text: str) -> str:
        return ''.join(CharTable.convert_char(c) for c in text)

    @staticmethod
    def normalize_chars(chars: List[str]) -> List[str]:
        return [CharTable.convert_char(c) for c in chars]

    @staticmethod
    def _init():
        with open(get_resource(HANLP_CHAR_TABLE), encoding='utf-8') as src:
            for line in src:
                cells = line.rstrip('\n')
                if len(cells) != 3:
                    continue
                a, _, b = cells
                CharTable.convert[a] = b
