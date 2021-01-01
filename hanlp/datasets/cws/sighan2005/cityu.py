# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:42
from hanlp.datasets.cws.sighan2005 import SIGHAN2005, make

SIGHAN2005_CITYU_DICT = SIGHAN2005 + "#" + "gold/cityu_training_words.utf8"
'''Dictionary built on trainings set.'''
SIGHAN2005_CITYU_TRAIN_ALL = SIGHAN2005 + "#" + "training/cityu_training.utf8"
'''Full training set.'''
SIGHAN2005_CITYU_TRAIN = SIGHAN2005 + "#" + "training/cityu_training_90.txt"
'''Training set (first 90% of the full official training set).'''
SIGHAN2005_CITYU_DEV = SIGHAN2005 + "#" + "training/cityu_training_10.txt"
'''Dev set (last 10% of full official training set).'''
SIGHAN2005_CITYU_TEST_INPUT = SIGHAN2005 + "#" + "testing/cityu_test.utf8"
'''Test input.'''
SIGHAN2005_CITYU_TEST = SIGHAN2005 + "#" + "gold/cityu_test_gold.utf8"
'''Test set.'''

make(SIGHAN2005_CITYU_TRAIN)
