# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 18:44
from hanlp_common.constant import HANLP_URL

_CTB_HOME = HANLP_URL + 'embeddings/SUDA-LA-CIP_20200109_021624.zip#'

_CTB5_DEP_HOME = _CTB_HOME + 'BPNN/data/ctb5/'

CTB5_DEP_TRAIN = _CTB5_DEP_HOME + 'train.conll'
'''Training set for ctb5 dependency parsing.'''
CTB5_DEP_DEV = _CTB5_DEP_HOME + 'dev.conll'
'''Dev set for ctb5 dependency parsing.'''
CTB5_DEP_TEST = _CTB5_DEP_HOME + 'test.conll'
'''Test set for ctb5 dependency parsing.'''

CIP_W2V_100_CN = _CTB_HOME + 'BPNN/data/embed.txt'
