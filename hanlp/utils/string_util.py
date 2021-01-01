# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-25 00:19
import unicodedata
from typing import List, Dict


def format_scores(results: Dict[str, float]) -> str:
    return ' - '.join(f'{k}: {v:.4f}' for (k, v) in results.items())


def ispunct(token):
    return all(unicodedata.category(char).startswith('P')
               for char in token)


def split_long_sentence_into(tokens: List[str], max_seq_length, sent_delimiter=None, char_level=False,
                             hard_constraint=False):
    punct_offset = [i for i, x in enumerate(tokens) if
                    ((sent_delimiter and x in sent_delimiter) or (not sent_delimiter and ispunct(x)))]
    if not punct_offset:
        # treat every token as punct
        punct_offset = [i for i in range(len(tokens))]
    punct_offset += [len(tokens)]
    token_to_char_offset = []
    if char_level:
        offset = 0
        for token in tokens:
            token_to_char_offset.append(offset)
            offset += len(token)
        token_to_char_offset.append(offset)

    start = 0
    for i, offset in enumerate(punct_offset[:-1]):
        end = punct_offset[i + 1]
        length_at_next_punct = _len(start, end, token_to_char_offset, char_level)
        if length_at_next_punct >= max_seq_length:
            if hard_constraint:
                yield from _gen_short_sent(tokens, start, offset, max_seq_length, token_to_char_offset, char_level)
            else:
                yield tokens[start: offset + 1]
            start = offset + 1
    offset = punct_offset[-1]
    if start < offset:
        offset -= 1
        length_at_next_punct = _len(start, offset, token_to_char_offset, char_level)
        if length_at_next_punct >= max_seq_length and hard_constraint:
            yield from _gen_short_sent(tokens, start, offset, max_seq_length, token_to_char_offset, char_level)
        else:
            yield tokens[start:]


def _gen_short_sent(tokens, start, offset, max_seq_length, token_to_char_offset, char_level):
    while start <= offset:
        for j in range(offset + 1, start, -1):
            if _len(start, j, token_to_char_offset, char_level) <= max_seq_length or j == start + 1:
                yield tokens[start: j]
                start = j
                break


def _len(start, end, token_to_char_offset, char_level):
    if char_level:
        length_at_next_punct = token_to_char_offset[end] - token_to_char_offset[start]
    else:
        length_at_next_punct = end - start
    return length_at_next_punct


def guess_delimiter(tokens):
    if all(ord(c) < 128 for c in ''.join(tokens)):
        delimiter_in_entity = ' '
    else:
        delimiter_in_entity = ''
    return delimiter_in_entity


def split_long_sent(sent, delimiters, max_seq_length):
    parts = []
    offset = 0
    for idx, char in enumerate(sent):
        if char in delimiters:
            parts.append(sent[offset:idx + 1])
            offset = idx + 1
    if not parts:
        yield sent
        return
    short = []
    for idx, part in enumerate(parts):
        short += part
        if idx == len(parts) - 1:
            yield short
        else:
            if len(short) + len(parts[idx + 1]) > max_seq_length:
                yield short
                short = []
