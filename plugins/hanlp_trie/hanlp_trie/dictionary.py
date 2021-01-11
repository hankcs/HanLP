# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-29 17:53
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Union, Sequence, Iterable, Optional

from hanlp_common.configurable import Configurable
from hanlp_common.reflection import classpath_of
from hanlp_trie.trie import Trie


class DictInterface(ABC):
    @abstractmethod
    def tokenize(self, text: Union[str, Sequence[str]]) -> List[Tuple[int, int, Any]]:
        """Implement this method to tokenize a piece of text into a list of non-intersect spans, each span is a tuple
        of ``(begin_offset, end_offset, label)``, where label is some properties related to this span and downstream
        tasks have the freedom to define what kind of labels they want.

        Args:
            text: The text to be tokenized.

        Returns:
              A list of tokens.

        """
        pass

    def split(self, text: Union[str, Sequence[str]]) -> List[Tuple[int, int, Any]]:
        """Like the :meth:`str.split`, this method splits a piece of text into chunks by taking the keys in this
        dictionary as delimiters. It performs longest-prefix-matching on text and split it whenever a longest key is
        matched. Unlike the :meth:`str.split`, it inserts matched keys into the results list right after where they are
        found. So that the text can be restored by joining chunks in the results list.

        Args:
            text: A piece of text.

        Returns:
            A list of chunks, each chunk is a span of ``(begin_offset, end_offset, label)``, where label is some
            properties related to this span and downstream tasks.
        """
        offset = 0
        spans = []
        for begin, end, label in self.tokenize(text):
            if begin > offset:
                spans.append(text[offset:begin])
            spans.append((begin, end, label))
            offset = end
        if offset < len(text):
            spans.append(text[offset:])
        return spans


class TrieDict(Trie, DictInterface, Configurable):
    def __init__(self, dictionary: Optional[Union[Dict[str, Any], Iterable[str]]] = None) -> None:
        r"""
        A dict-like structure for fast custom dictionary strategies in tokenization and tagging. It is built with
        a dict of key-value pairs or a set of strings. When a set is passed in, it will be turned into a dict where each
        key is assigned with a boolean value ``True``.

        Args:
            dictionary: A custom dictionary of string-value pairs.
        """
        super().__init__(dictionary)

    def tokenize(self, text: Union[str, Sequence[str]]) -> List[Tuple[int, int, Any]]:
        return self.parse_longest(text)

    def split_batch(self, data: List[str]) -> Tuple[List[str], List[int], List[List[Tuple[int, int, Any]]]]:
        """ A handy method to perform longest-prefix-matching on a batch of sentences. It tokenize each sentence, record
        the chunks being either a key in the dict or a span outside of the dict. The spans are then packed into a new
        batch and returned along with the following information:

            - which sentence a span belongs to
            - the matched keys along with their spans and values.

        This method bridges the gap between statistical models and rule-based gazetteers.
        It's used in conjunction with :meth:`~hanlp_trie.dictionary.TrieDict.merge_batch`.

        Args:
            data: A batch of sentences.

        Returns:
            A tuple of the new batch, the belonging information and the keys.
        """
        new_data, new_data_belongs, parts = [], [], []
        for idx, sent in enumerate(data):
            parts.append([])
            found = self.tokenize(sent)
            if found:
                pre_start = 0
                for start, end, info in found:
                    if start > pre_start:
                        new_data.append(sent[pre_start:start])
                        new_data_belongs.append(idx)
                    pre_start = end
                    parts[idx].append((start, end, info))
                if pre_start != len(sent):
                    new_data.append(sent[pre_start:])
                    new_data_belongs.append(idx)
            else:
                new_data.append(sent)
                new_data_belongs.append(idx)
        return new_data, new_data_belongs, parts

    @staticmethod
    def merge_batch(data, new_outputs, new_data_belongs, parts):
        """ A helper method to merge the outputs of split batch back by concatenating the output per span with the key
        used to split it. It's used in conjunction with :meth:`~hanlp_trie.dictionary.TrieDict.split_batch`.

        Args:
            data: Split batch.
            new_outputs: Outputs of the split batch.
            new_data_belongs: Belonging information.
            parts: The keys.

        Returns:
            Merged outputs.
        """
        outputs = []
        segments = []
        for idx in range(len(data)):
            segments.append([])
        for o, b in zip(new_outputs, new_data_belongs):
            dst = segments[b]
            dst.append(o)
        for s, p, sent in zip(segments, parts, data):
            s: list = s
            if p:
                dst = []
                offset = 0
                for start, end, info in p:
                    while offset < start:
                        head = s.pop(0)
                        offset += sum(len(token) for token in head)
                        dst += head
                    if isinstance(info, list):
                        dst += info
                    elif isinstance(info, str):
                        dst.append(info)
                    else:
                        dst.append(sent[start:end])
                    offset = end
                if s:
                    assert len(s) == 1
                    dst += s[0]
                outputs.append(dst)
            else:
                outputs.append(s[0])
        return outputs

    @property
    def config(self):
        return {
            'classpath': classpath_of(self),
            'dictionary': dict(self.items())
        }
