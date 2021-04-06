# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-12 20:34


def generate_words_per_line(file_path):
    with open(file_path, encoding='utf-8') as src:
        for line in src:
            cells = line.strip().split()
            if not cells:
                continue
            yield cells


def words_to_bmes(words):
    tags = []
    for w in words:
        if not w:
            raise ValueError('{} contains None or zero-length word {}'.format(str(words), w))
        if len(w) == 1:
            tags.append('S')
        else:
            tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
    return tags


def words_to_bi(words):
    tags = []
    for w in words:
        if not w:
            raise ValueError('{} contains None or zero-length word {}'.format(str(words), w))
        tags.extend(['B'] + ['I'] * (len(w) - 1))
    return tags


def bmes_to_words(chars, tags):
    result = []
    if len(chars) == 0:
        return result
    word = chars[0]

    for c, t in zip(chars[1:], tags[1:]):
        if t == 'B' or t == 'S':
            result.append(word)
            word = ''
        word += c
    if len(word) != 0:
        result.append(word)

    return result


def bmes_to_spans(tags):
    result = []
    offset = 0
    pre_offset = 0
    for t in tags[1:]:
        offset += 1
        if t == 'B' or t == 'S':
            result.append((pre_offset, offset))
            pre_offset = offset
    if offset != len(tags):
        result.append((pre_offset, len(tags)))

    return result


def bmes_of(sentence, segmented):
    if segmented:
        chars = []
        tags = []
        words = sentence.split()
        for w in words:
            chars.extend(list(w))
            if len(w) == 1:
                tags.append('S')
            else:
                tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
    else:
        chars = list(sentence)
        tags = ['S'] * len(chars)
    return chars, tags


def iobes_to_bilou(src, dst):
    with open(src) as src, open(dst, 'w') as out:
        for line in src:
            line = line.strip()
            if not line:
                out.write('\n')
                continue
            word, tag = line.split('\t')
            if tag.startswith('E-'):
                tag = 'L-' + tag[2:]
            elif tag.startswith('S-'):
                tag = 'U-' + tag[2:]
            out.write(f'{word}\t{tag}\n')
