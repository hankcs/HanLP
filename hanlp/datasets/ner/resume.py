# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-08 12:10
from hanlp.common.dataset import TransformableDataset

from hanlp.utils.io_util import get_resource, generate_words_tags_from_tsv

_RESUME_NER_HOME = 'https://github.com/jiesutd/LatticeLSTM/archive/master.zip#'

RESUME_NER_TRAIN = _RESUME_NER_HOME + 'ResumeNER/train.char.bmes'
'''Training set of Resume in char level.'''
RESUME_NER_DEV = _RESUME_NER_HOME + 'ResumeNER/dev.char.bmes'
'''Dev set of Resume in char level.'''
RESUME_NER_TEST = _RESUME_NER_HOME + 'ResumeNER/test.char.bmes'
'''Test set of Resume in char level.'''

