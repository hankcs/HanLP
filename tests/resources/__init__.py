# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-24 23:03
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def res(path_to_project_root: str) -> str:
    return os.path.join(project_root, path_to_project_root)


sample_word2vec = res('tests/resources/sample_word2vec.txt')

ptb_pos_train = res('tests/resources/ptb_pos/train.debug.tsv')
ptb_pos_dev = res('tests/resources/ptb_pos/dev.debug.tsv')
ptb_pos_test = res('tests/resources/ptb_pos/test.debug.tsv')
