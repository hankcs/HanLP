# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-03 23:33
from hanlp.common.dataset import TransformableDataset

from hanlp.utils.io_util import get_resource, generate_words_tags_from_tsv

_WEIBO_NER_HOME = 'https://github.com/hltcoe/golden-horse/archive/master.zip#data/'

WEIBO_NER_TRAIN = _WEIBO_NER_HOME + 'weiboNER_2nd_conll.train'
'''Training set of Weibo in char level.'''
WEIBO_NER_DEV = _WEIBO_NER_HOME + 'weiboNER_2nd_conll.dev'
'''Dev set of Weibo in char level.'''
WEIBO_NER_TEST = _WEIBO_NER_HOME + 'weiboNER_2nd_conll.test'
'''Test set of Weibo in char level.'''
