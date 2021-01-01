# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-10-14 20:54

from hanlp.datasets.parsing._ctb_utils import make_ctb

_CTB8_HOME = 'https://wakespace.lib.wfu.edu/bitstream/handle/10339/39379/LDC2013T21.tgz#data/'

CTB8_CWS_TRAIN = _CTB8_HOME + 'tasks/cws/train.txt'
'''Training set for ctb8 Chinese word segmentation.'''
CTB8_CWS_DEV = _CTB8_HOME + 'tasks/cws/dev.txt'
'''Dev set for ctb8 Chinese word segmentation.'''
CTB8_CWS_TEST = _CTB8_HOME + 'tasks/cws/test.txt'
'''Test set for ctb8 Chinese word segmentation.'''

CTB8_POS_TRAIN = _CTB8_HOME + 'tasks/pos/train.tsv'
'''Training set for ctb8 PoS tagging.'''
CTB8_POS_DEV = _CTB8_HOME + 'tasks/pos/dev.tsv'
'''Dev set for ctb8 PoS tagging.'''
CTB8_POS_TEST = _CTB8_HOME + 'tasks/pos/test.tsv'
'''Test set for ctb8 PoS tagging.'''

CTB8_BRACKET_LINE_TRAIN = _CTB8_HOME + 'tasks/par/train.txt'
'''Training set for ctb8 constituency parsing with empty categories.'''
CTB8_BRACKET_LINE_DEV = _CTB8_HOME + 'tasks/par/dev.txt'
'''Dev set for ctb8 constituency parsing with empty categories.'''
CTB8_BRACKET_LINE_TEST = _CTB8_HOME + 'tasks/par/test.txt'
'''Test set for ctb8 constituency parsing with empty categories.'''

CTB8_BRACKET_LINE_NOEC_TRAIN = _CTB8_HOME + 'tasks/par/train.noempty.txt'
'''Training set for ctb8 constituency parsing without empty categories.'''
CTB8_BRACKET_LINE_NOEC_DEV = _CTB8_HOME + 'tasks/par/dev.noempty.txt'
'''Dev set for ctb8 constituency parsing without empty categories.'''
CTB8_BRACKET_LINE_NOEC_TEST = _CTB8_HOME + 'tasks/par/test.noempty.txt'
'''Test set for ctb8 constituency parsing without empty categories.'''

CTB8_SD330_TRAIN = _CTB8_HOME + 'tasks/dep/train.conllx'
'''Training set for ctb8 in Stanford Dependencies 3.3.0 standard.'''
CTB8_SD330_DEV = _CTB8_HOME + 'tasks/dep/dev.conllx'
'''Dev set for ctb8 in Stanford Dependencies 3.3.0 standard.'''
CTB8_SD330_TEST = _CTB8_HOME + 'tasks/dep/test.conllx'
'''Test set for ctb8 in Stanford Dependencies 3.3.0 standard.'''

make_ctb(_CTB8_HOME)
