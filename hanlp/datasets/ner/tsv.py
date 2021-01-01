# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-24 23:09
from typing import Union, List, Callable

from hanlp.common.dataset import TransformableDataset
from hanlp.utils.io_util import get_resource, generate_words_tags_from_tsv
from hanlp.utils.string_util import split_long_sentence_into


class TSVTaggingDataset(TransformableDataset):

    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None,
                 max_seq_len=None,
                 sent_delimiter=None,
                 char_level=False,
                 hard_constraint=False,
                 **kwargs
                 ) -> None:
        """

        Args:
            data: The local or remote path to a dataset, or a list of samples where each sample is a dict.
            transform: Predefined transform(s).
            cache: ``True`` to enable caching, so that transforms won't be called twice.
            generate_idx: Create a :const:`~hanlp_common.constants.IDX` field for each sample to store its order in dataset. Useful for prediction when
                samples are re-ordered by a sampler.
            max_seq_len: Sentences longer than ``max_seq_len`` will be split into shorter ones if possible.
            sent_delimiter: Delimiter between sentences, like period or comma, which indicates a long sentence can
                be split here.
            char_level: Whether the sequence length is measured at char level, which is never the case for
                lemmatization.
            hard_constraint: Whether to enforce hard length constraint on sentences. If there is no ``sent_delimiter``
                in a sentence, it will be split at a token anyway.
            kwargs: Not used.
        """
        self.char_level = char_level
        self.hard_constraint = hard_constraint
        self.sent_delimiter = sent_delimiter
        self.max_seq_len = max_seq_len
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath):
        """Load a ``.tsv`` file. A ``.tsv`` file for tagging is defined as a tab separated text file, where non-empty
        lines have two columns for token and tag respectively, empty lines mark the end of sentences.

        Args:
            filepath: Path to a ``.tsv`` tagging file.

        .. highlight:: bash
        .. code-block:: bash

            $ head eng.train.tsv
            -DOCSTART-      O

            EU      S-ORG
            rejects O
            German  S-MISC
            call    O
            to      O
            boycott O
            British S-MISC
            lamb    O

        """
        filepath = get_resource(filepath)
        # idx = 0
        for words, tags in generate_words_tags_from_tsv(filepath, lower=False):
            # idx += 1
            # if idx % 1000 == 0:
            #     print(f'\rRead instances {idx // 1000}k', end='')
            if self.max_seq_len:
                start = 0
                for short_sents in split_long_sentence_into(words, self.max_seq_len, self.sent_delimiter,
                                                            char_level=self.char_level,
                                                            hard_constraint=self.hard_constraint):
                    end = start + len(short_sents)
                    yield {'token': short_sents, 'tag': tags[start:end]}
                    start = end
            else:
                yield {'token': words, 'tag': tags}
        # print('\r', end='')
