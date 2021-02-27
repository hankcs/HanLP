# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-01 12:35
from typing import Union, List, Callable

from hanlp.common.dataset import TransformableDataset
from hanlp.utils.io_util import TimingFileIterator
from hanlp.utils.span_util import words_to_bmes, words_to_bi
from hanlp.utils.string_util import split_long_sentence_into


class TextTokenizingDataset(TransformableDataset):
    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None,
                 delimiter=None,
                 max_seq_len=None,
                 sent_delimiter=None,
                 char_level=False,
                 hard_constraint=False,
                 ) -> None:
        """A dataset for tagging tokenization tasks.

        Args:
            data: The local or remote path to a dataset, or a list of samples where each sample is a dict.
            transform: Predefined transform(s).
            cache: ``True`` to enable caching, so that transforms won't be called twice.
            generate_idx: Create a :const:`~hanlp_common.constants.IDX` field for each sample to store its order in dataset. Useful for prediction when
                samples are re-ordered by a sampler.
            delimiter: Delimiter between tokens used to split a line in the corpus.
            max_seq_len: Sentences longer than ``max_seq_len`` will be split into shorter ones if possible.
            sent_delimiter: Delimiter between sentences, like period or comma, which indicates a long sentence can
                be split here.
            char_level: Whether the sequence length is measured at char level.
            hard_constraint: Whether to enforce hard length constraint on sentences. If there is no ``sent_delimiter``
                in a sentence, it will be split at a token anyway.
        """
        self.hard_constraint = hard_constraint
        self.char_level = char_level
        self.sent_delimiter = sent_delimiter
        self.max_seq_len = max_seq_len
        self.delimiter = delimiter
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath: str):
        """Load tokenized corpus. The format is one sentence per line, where each line consisits of tokens seperated
        by a delimiter (usually space).

        .. highlight:: bash
        .. code-block:: bash

            $ head train.txt
            上海 浦东 开发 与 法制 建设 同步
            新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）

        Args:
            filepath: The path to the corpus.
        """
        f = TimingFileIterator(filepath)
        # longest_sent = 0
        for line in f:
            line = line.rstrip('\n')
            tokens = line.split(self.delimiter)
            if not tokens:
                continue
            if self.max_seq_len and sum(len(t) for t in tokens) > self.max_seq_len:
                # debug = []
                for short_sents in split_long_sentence_into(tokens, self.max_seq_len, self.sent_delimiter,
                                                            char_level=self.char_level,
                                                            hard_constraint=self.hard_constraint):
                    # debug.extend(short_sents)
                    # longest_sent = max(longest_sent, len(''.join(short_sents)))
                    yield {'token': short_sents}
                # assert debug == tokens
            else:
                # longest_sent = max(longest_sent, len(''.join(tokens)))
                yield {'token': tokens}
            f.log(line[:20])
        f.erase()
        # print(f'Longest sent: {longest_sent} in {filepath}')


def generate_tags_for_subtokens(sample: dict, tagging_scheme='BMES'):
    # We could use token_token_span but we don't want token_token_span in the batch
    subtokens_group = sample.get('token_subtoken_offsets_group', None)
    sample['raw_token'] = sample['token']
    sample['token'] = offsets_to_subtokens(sample.get('token_') or sample['token'], sample['token_subtoken_offsets'],
                                           subtokens_group)
    if subtokens_group:
        if tagging_scheme == 'BMES':
            sample['tag'] = words_to_bmes(subtokens_group)
        elif tagging_scheme == 'BI':
            sample['tag'] = words_to_bi(subtokens_group)
        else:
            raise NotImplementedError(f'Unsupported tagging scheme {tagging_scheme}.')
    return sample


def offsets_to_subtokens(tokens, token_subtoken_offsets, token_input_tokens_group):
    results = []
    if token_input_tokens_group:
        for subtokens, token in zip(token_input_tokens_group, tokens):
            for b, e in subtokens:
                results.append(token[b:e])
    else:
        offset = -1  # BERT produces 'ᄒ', '##ᅡ', '##ᆫ' for '한' and they share the same span
        for b, e in token_subtoken_offsets:
            if b < offset:
                continue
            offset = e
            results.append(tokens[b:e])
    return results
