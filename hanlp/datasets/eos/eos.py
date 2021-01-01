# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-26 18:12
import itertools
from collections import Counter
from typing import Union, List, Callable

from hanlp.common.dataset import TransformableDataset
from hanlp.utils.io_util import TimingFileIterator
from hanlp.utils.log_util import cprint
from hanlp.utils.string_util import ispunct


class SentenceBoundaryDetectionDataset(TransformableDataset):

    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 append_after_sentence=None,
                 eos_chars=None,
                 eos_char_min_freq=200,
                 eos_char_is_punct=True,
                 window_size=5,
                 **kwargs,
                 ) -> None:
        """Dataset for sentence boundary detection (eos).

        Args:
            data: The local or remote path to a dataset, or a list of samples where each sample is a dict.
            transform: Predefined transform(s).
            cache: ``True`` to enable caching, so that transforms won't be called twice.
            append_after_sentence: A :class:`str` to insert at the tail of each sentence. For example, English always
                have a space between sentences.
            eos_chars: Punctuations at the tail of sentences. If ``None``, then it will built from training samples.
            eos_char_min_freq: Minimal frequency to keep a eos char.
            eos_char_is_punct: Limit eos chars to punctuations.
            window_size: Window size to extract ngram features.
            kwargs: Not used.
        """
        self.eos_char_is_punct = eos_char_is_punct
        self.append_after_sentence = append_after_sentence
        self.window_size = window_size
        self.eos_chars = eos_chars
        self.eos_char_min_freq = eos_char_min_freq
        super().__init__(data, transform, cache)

    def load_file(self, filepath: str):
        """Load eos corpus.

        Args:
            filepath: Path to the corpus.

        .. highlight:: bash
        .. code-block:: bash

            $ head -n 2 ctb8.txt
            中国经济简讯
            新华社北京十月二十九日电中国经济简讯

        """
        f = TimingFileIterator(filepath)
        sents = []
        eos_offsets = []
        offset = 0
        for line in f:
            if not line.strip():
                continue
            line = line.rstrip('\n')
            eos_offsets.append(offset + len(line.rstrip()) - 1)
            offset += len(line)
            if self.append_after_sentence:
                line += self.append_after_sentence
                offset += len(self.append_after_sentence)
            f.log(line)
            sents.append(line)
        f.erase()
        corpus = list(itertools.chain.from_iterable(sents))

        if self.eos_chars:
            if not isinstance(self.eos_chars, set):
                self.eos_chars = set(self.eos_chars)
        else:
            eos_chars = Counter()
            for i in eos_offsets:
                eos_chars[corpus[i]] += 1
            self.eos_chars = set(k for (k, v) in eos_chars.most_common() if
                                 v >= self.eos_char_min_freq and (not self.eos_char_is_punct or ispunct(k)))
            cprint(f'eos_chars = [yellow]{self.eos_chars}[/yellow]')

        eos_index = 0
        eos_offsets = [i for i in eos_offsets if corpus[i] in self.eos_chars]
        window_size = self.window_size
        for i, c in enumerate(corpus):
            if c in self.eos_chars:
                window = corpus[i - window_size: i + window_size + 1]
                label_id = 1. if eos_offsets[eos_index] == i else 0.
                if label_id > 0:
                    eos_index += 1
                yield {'char': window, 'label_id': label_id}
        assert eos_index == len(eos_offsets), f'{eos_index} != {len(eos_offsets)}'
