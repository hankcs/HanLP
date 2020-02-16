# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-21 20:39
import os

from hanlp.datasets.parsing.ud import concat_treebanks
from hanlp.datasets.parsing.ud.ud27 import _UD_27_HOME

_UD_27_MULTILINGUAL_HOME = concat_treebanks(_UD_27_HOME, '2.7')
UD_27_MULTILINGUAL_TRAIN = os.path.join(_UD_27_MULTILINGUAL_HOME, 'train.conllu')
"Training set of multilingual UD_27 obtained by concatenating all training sets."
UD_27_MULTILINGUAL_DEV = os.path.join(_UD_27_MULTILINGUAL_HOME, 'dev.conllu')
"Dev set of multilingual UD_27 obtained by concatenating all dev sets."
UD_27_MULTILINGUAL_TEST = os.path.join(_UD_27_MULTILINGUAL_HOME, 'test.conllu')
"Test set of multilingual UD_27 obtained by concatenating all test sets."
