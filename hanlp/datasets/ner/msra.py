# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 23:13

_MSRA_NER_HOME = 'http://file.hankcs.com/corpus/msra_ner.zip'
_MSRA_NER_TOKEN_LEVEL_HOME = 'http://file.hankcs.com/corpus/msra_ner_token_level.zip'

MSRA_NER_CHAR_LEVEL_TRAIN = f'{_MSRA_NER_HOME}#train.tsv'
'''Training set of MSRA (:cite:`levow-2006-third`) in character level.'''
MSRA_NER_CHAR_LEVEL_DEV = f'{_MSRA_NER_HOME}#dev.tsv'
'''Dev set of MSRA (:cite:`levow-2006-third`) in character level.'''
MSRA_NER_CHAR_LEVEL_TEST = f'{_MSRA_NER_HOME}#test.tsv'
'''Test set of MSRA (:cite:`levow-2006-third`) in character level.'''

MSRA_NER_TOKEN_LEVEL_IOBES_TRAIN = f'{_MSRA_NER_TOKEN_LEVEL_HOME}#word_level.train.tsv'
'''Training set of MSRA (:cite:`levow-2006-third`) in token level.'''
MSRA_NER_TOKEN_LEVEL_IOBES_DEV = f'{_MSRA_NER_TOKEN_LEVEL_HOME}#word_level.dev.tsv'
'''Dev set of MSRA (:cite:`levow-2006-third`) in token level.'''
MSRA_NER_TOKEN_LEVEL_IOBES_TEST = f'{_MSRA_NER_TOKEN_LEVEL_HOME}#word_level.test.tsv'
'''Test set of MSRA (:cite:`levow-2006-third`) in token level.'''

MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_TRAIN = f'{_MSRA_NER_TOKEN_LEVEL_HOME}#word_level.train.short.tsv'
'''Training set of shorten (<= 128 tokens) MSRA (:cite:`levow-2006-third`) in token level.'''
MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_DEV = f'{_MSRA_NER_TOKEN_LEVEL_HOME}#word_level.dev.short.tsv'
'''Dev set of shorten (<= 128 tokens) MSRA (:cite:`levow-2006-third`) in token level.'''
MSRA_NER_TOKEN_LEVEL_SHORT_IOBES_TEST = f'{_MSRA_NER_TOKEN_LEVEL_HOME}#word_level.test.short.tsv'
'''Test set of shorten (<= 128 tokens) MSRA (:cite:`levow-2006-third`) in token level.'''

MSRA_NER_TOKEN_LEVEL_SHORT_JSON_TRAIN = f'{_MSRA_NER_TOKEN_LEVEL_HOME}#word_level.train.short.jsonlines'
'''Training set of shorten (<= 128 tokens) MSRA (:cite:`levow-2006-third`) in token level and jsonlines format.'''
MSRA_NER_TOKEN_LEVEL_SHORT_JSON_DEV = f'{_MSRA_NER_TOKEN_LEVEL_HOME}#word_level.dev.short.jsonlines'
'''Dev set of shorten (<= 128 tokens) MSRA (:cite:`levow-2006-third`) in token level and jsonlines format.'''
MSRA_NER_TOKEN_LEVEL_SHORT_JSON_TEST = f'{_MSRA_NER_TOKEN_LEVEL_HOME}#word_level.test.short.jsonlines'
'''Test set of shorten (<= 128 tokens) MSRA (:cite:`levow-2006-third`) in token level and jsonlines format.'''
