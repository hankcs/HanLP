# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-25 00:19
import unicodedata
from typing import List, Dict

import tensorflow as tf


def format_metrics(metrics: List[tf.keras.metrics.Metric]):
    return ' - '.join(f'{m.name}: {m.result():.4f}' for m in metrics)


def format_scores(results: Dict[str, float]) -> str:
    return ' - '.join(f'{k}: {v:.4f}' for (k, v) in results.items())


def ispunct(token):
    return all(unicodedata.category(char).startswith('P')
               for char in token)


def split_long_sentence_into(tokens: List[str], max_seq_length):
    punct_offset = [i for i, x in enumerate(tokens) if ispunct(x)]
    if not punct_offset:
        # treat every token as punct
        punct_offset = [i for i in range(len(tokens))]
    punct_offset += [len(tokens)]
    start = 0
    for i, offset in enumerate(punct_offset[:-1]):
        if punct_offset[i + 1] - start >= max_seq_length:
            yield tokens[start: offset + 1]
            start = offset + 1
    if start < punct_offset[-1]:
        yield tokens[start:]


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