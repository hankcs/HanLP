# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:42
from hanlp.datasets.cws.sighan2005 import SIGHAN2005, make

SIGHAN2005_PKU_DICT = SIGHAN2005 + "#" + "gold/pku_training_words.utf8"
SIGHAN2005_PKU_TRAIN_FULL = SIGHAN2005 + "#" + "training/pku_training.utf8"
SIGHAN2005_PKU_TRAIN = SIGHAN2005 + "#" + "training/pku_training_90.txt"
SIGHAN2005_PKU_VALID = SIGHAN2005 + "#" + "training/pku_training_10.txt"
SIGHAN2005_PKU_TEST_INPUT = SIGHAN2005 + "#" + "testing/pku_test.utf8"
SIGHAN2005_PKU_TEST = SIGHAN2005 + "#" + "gold/pku_test_gold.utf8"

make(SIGHAN2005_PKU_TRAIN)
