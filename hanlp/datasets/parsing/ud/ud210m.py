# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-21 20:39
import os

from hanlp.datasets.parsing.ud import concat_treebanks
from hanlp.datasets.parsing.ud.ud210 import _UD_210_HOME

_UD_210_MULTILINGUAL_HOME = concat_treebanks(_UD_210_HOME, '2.10')
UD_210_MULTILINGUAL_TRAIN = os.path.join(_UD_210_MULTILINGUAL_HOME, 'train.conllu')
"Training set of multilingual UD_210 obtained by concatenating all training sets."
UD_210_MULTILINGUAL_DEV = os.path.join(_UD_210_MULTILINGUAL_HOME, 'dev.conllu')
"Dev set of multilingual UD_210 obtained by concatenating all dev sets."
UD_210_MULTILINGUAL_TEST = os.path.join(_UD_210_MULTILINGUAL_HOME, 'test.conllu')
"Test set of multilingual UD_210 obtained by concatenating all test sets."
