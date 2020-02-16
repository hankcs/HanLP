# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-09 19:07
from typing import List

from hanlp.utils.io_util import get_resource
from hanlp_common.io import load_json

HANLP_CHAR_TABLE_TXT = 'https://file.hankcs.com/corpus/char_table.zip#CharTable.txt'
HANLP_CHAR_TABLE_JSON = 'https://file.hankcs.com/corpus/char_table.json.zip'


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
        CharTable.convert = CharTable.load()

    @staticmethod
    def load():
        mapper = {}
        with open(get_resource(HANLP_CHAR_TABLE_TXT), encoding='utf-8') as src:
            for line in src:
                cells = line.rstrip('\n')
                if len(cells) != 3:
                    continue
                a, _, b = cells
                mapper[a] = b
        return mapper


class JsonCharTable(CharTable):

    @staticmethod
    def load():
        return load_json(get_resource(HANLP_CHAR_TABLE_JSON))


