# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:42
from hanlp.datasets.cws.sighan2005 import SIGHAN2005, make

SIGHAN2005_MSR_DICT = SIGHAN2005 + "#" + "gold/msr_training_words.utf8"
SIGHAN2005_MSR_TRAIN_FULL = SIGHAN2005 + "#" + "training/msr_training.utf8"
SIGHAN2005_MSR_TRAIN = SIGHAN2005 + "#" + "training/msr_training_90.txt"
SIGHAN2005_MSR_VALID = SIGHAN2005 + "#" + "training/msr_training_10.txt"
SIGHAN2005_MSR_TEST_INPUT = SIGHAN2005 + "#" + "testing/msr_test.utf8"
SIGHAN2005_MSR_TEST = SIGHAN2005 + "#" + "gold/msr_test_gold.utf8"

make(SIGHAN2005_MSR_TRAIN)
