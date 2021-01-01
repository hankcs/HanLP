# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-10-14 20:54
from urllib.error import HTTPError

from hanlp.datasets.parsing._ctb_utils import make_ctb
from hanlp.utils.io_util import get_resource, path_from_url

_CTB9_HOME = 'https://catalog.ldc.upenn.edu/LDC2016T13/ctb9.0_LDC2016T13.tgz#data/'

CTB9_CWS_TRAIN = _CTB9_HOME + 'tasks/cws/train.txt'
'''Training set for ctb9 Chinese word segmentation.'''
CTB9_CWS_DEV = _CTB9_HOME + 'tasks/cws/dev.txt'
'''Dev set for ctb9 Chinese word segmentation.'''
CTB9_CWS_TEST = _CTB9_HOME + 'tasks/cws/test.txt'
'''Test set for ctb9 Chinese word segmentation.'''

CTB9_POS_TRAIN = _CTB9_HOME + 'tasks/pos/train.tsv'
'''Training set for ctb9 PoS tagging.'''
CTB9_POS_DEV = _CTB9_HOME + 'tasks/pos/dev.tsv'
'''Dev set for ctb9 PoS tagging.'''
CTB9_POS_TEST = _CTB9_HOME + 'tasks/pos/test.tsv'
'''Test set for ctb9 PoS tagging.'''

CTB9_BRACKET_LINE_TRAIN = _CTB9_HOME + 'tasks/par/train.txt'
'''Training set for ctb9 constituency parsing with empty categories.'''
CTB9_BRACKET_LINE_DEV = _CTB9_HOME + 'tasks/par/dev.txt'
'''Dev set for ctb9 constituency parsing with empty categories.'''
CTB9_BRACKET_LINE_TEST = _CTB9_HOME + 'tasks/par/test.txt'
'''Test set for ctb9 constituency parsing with empty categories.'''

CTB9_BRACKET_LINE_NOEC_TRAIN = _CTB9_HOME + 'tasks/par/train.noempty.txt'
'''Training set for ctb9 constituency parsing without empty categories.'''
CTB9_BRACKET_LINE_NOEC_DEV = _CTB9_HOME + 'tasks/par/dev.noempty.txt'
'''Dev set for ctb9 constituency parsing without empty categories.'''
CTB9_BRACKET_LINE_NOEC_TEST = _CTB9_HOME + 'tasks/par/test.noempty.txt'
'''Test set for ctb9 constituency parsing without empty categories.'''

CTB9_SD330_TRAIN = _CTB9_HOME + 'tasks/dep/train.conllx'
'''Training set for ctb9 in Stanford Dependencies 3.3.0 standard.'''
CTB9_SD330_DEV = _CTB9_HOME + 'tasks/dep/dev.conllx'
'''Dev set for ctb9 in Stanford Dependencies 3.3.0 standard.'''
CTB9_SD330_TEST = _CTB9_HOME + 'tasks/dep/test.conllx'
'''Test set for ctb9 in Stanford Dependencies 3.3.0 standard.'''

try:
    get_resource(_CTB9_HOME)
except HTTPError:
    raise FileNotFoundError(
        'Chinese Treebank 9.0 is a copyright dataset owned by LDC which we cannot re-distribute. '
        f'Please apply for a licence from LDC (https://catalog.ldc.upenn.edu/LDC2016T13) '
        f'then download it to {path_from_url(_CTB9_HOME)}'
    )

make_ctb(_CTB9_HOME)
