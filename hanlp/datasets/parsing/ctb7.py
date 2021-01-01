# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 18:44
from hanlp.datasets.parsing.ctb5 import _CTB_HOME

_CTB7_HOME = _CTB_HOME + 'BPNN/data/ctb7/'

CTB7_DEP_TRAIN = _CTB7_HOME + 'train.conll'
'''Training set for ctb7 dependency parsing.'''
CTB7_DEP_DEV = _CTB7_HOME + 'dev.conll'
'''Dev set for ctb7 dependency parsing.'''
CTB7_DEP_TEST = _CTB7_HOME + 'test.conll'
'''Test set for ctb7 dependency parsing.'''
