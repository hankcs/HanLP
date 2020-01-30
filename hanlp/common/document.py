# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 04:16
import json
from typing import List

from hanlp.common.structure import SerializableDict
from hanlp.components.parsers.conll import CoNLLSentence
from hanlp.utils.util import collapse_json


class Sentence(SerializableDict):
    KEY_WORDS = 'words'
    KEY_POS = 'pos'
    KEY_NER = 'ner'

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.update(kwargs)

    @property
    def words(self) -> List[str]:
        return self.get(Sentence.KEY_WORDS)

    @words.setter
    def words(self, words: List[str]):
        self[Sentence.KEY_WORDS] = words


class Document(SerializableDict):
    def __init__(self) -> None:
        super().__init__()
        # self.sentences = []
        # self.tokens = []
        # self.part_of_speech_tags = []
        # self.named_entities = []
        # self.syntactic_dependencies = []
        # self.semantic_dependencies = []

    def __missing__(self, key):
        value = []
        self[key] = value
        return value

    def to_dict(self) -> dict:
        return dict((k, v) for k, v in self.items() if v)

    def to_json(self, ensure_ascii=False, indent=2) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)
        text = collapse_json(text, 4)
        return text

    def __str__(self) -> str:
        return self.to_json()

    def to_conll(self) -> List[CoNLLSentence]:
        # try to find if any field is conll type
        if self.semantic_dependencies and isinstance(self.semantic_dependencies[0], CoNLLSentence):
            return self.semantic_dependencies
        if self.syntactic_dependencies and isinstance(self.syntactic_dependencies[0], CoNLLSentence):
            return self.syntactic_dependencies
        for k, v in self.items():
            if len(v) and isinstance(v[0], CoNLLSentence):
                return v
