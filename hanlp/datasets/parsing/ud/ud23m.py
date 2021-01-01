# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-21 20:39
import os

from hanlp.datasets.parsing.ud import concat_treebanks
from .ud23 import _UD_23_HOME

_UD_23_MULTILINGUAL_HOME = concat_treebanks(_UD_23_HOME, '2.3')
UD_23_MULTILINGUAL_TRAIN = os.path.join(_UD_23_MULTILINGUAL_HOME, 'train.conllu')
UD_23_MULTILINGUAL_DEV = os.path.join(_UD_23_MULTILINGUAL_HOME, 'dev.conllu')
UD_23_MULTILINGUAL_TEST = os.path.join(_UD_23_MULTILINGUAL_HOME, 'test.conllu')
