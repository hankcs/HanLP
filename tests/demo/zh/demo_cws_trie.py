# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 21:25
from hanlp.common.trie import Trie

import hanlp

tokenizer = hanlp.load('LARGE_ALBERT_BASE')
text = 'NLP统计模型没有加规则，聪明人知道自己加。英文、数字、自定义词典统统都是规则。'
print(tokenizer(text))

trie = Trie()
trie.update({'自定义词典': 'custom_dict', '聪明人': 'smart'})


def split_sents(text: str, trie: Trie):
    words = trie.parse_longest(text)
    sents = []
    pre_start = 0
    offsets = []
    for word, value, start, end in words:
        if pre_start != start:
            sents.append(text[pre_start: start])
            offsets.append(pre_start)
        pre_start = end
    if pre_start != len(text):
        sents.append(text[pre_start:])
        offsets.append(pre_start)
    return sents, offsets, words


print(split_sents(text, trie))


def merge_parts(parts, offsets, words):
    items = [(i, p) for (i, p) in zip(offsets, parts)]
    items += [(start, [word]) for (word, value, start, end) in words]
    # In case you need the tag, use the following line instead
    # items += [(start, [(word, value)]) for (word, value, start, end) in words]
    return [each for x in sorted(items) for each in x[1]]


tokenizer = hanlp.pipeline() \
    .append(split_sents, output_key=('parts', 'offsets', 'words'), trie=trie) \
    .append(tokenizer, input_key='parts', output_key='tokens') \
    .append(merge_parts, input_key=('tokens', 'offsets', 'words'), output_key='merged')

print(tokenizer(text))
