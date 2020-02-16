# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-25 18:48
import glob
import os

from hanlp.utils.io_util import get_resource, merge_files

_CONLL2012_EN_HOME = 'https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO/archive/master.zip#conll-formatted-ontonotes-5.0/data'
# These are v4 of OntoNotes, in .conll format
CONLL2012_EN_TRAIN = _CONLL2012_EN_HOME + '/train/data/english/annotations'
CONLL2012_EN_DEV = _CONLL2012_EN_HOME + '/development/data/english/annotations'
CONLL2012_EN_TEST = _CONLL2012_EN_HOME + '/conll-2012-test/data/english/annotations'


def conll_2012_en_combined():
    home = get_resource(_CONLL2012_EN_HOME)
    outputs = ['train', 'dev', 'test']
    for i in range(len(outputs)):
        outputs[i] = f'{home}/conll12_en/{outputs[i]}.conll'
    if all(os.path.isfile(x) for x in outputs):
        return outputs
    os.makedirs(os.path.dirname(outputs[0]), exist_ok=True)
    for in_path, out_path in zip([CONLL2012_EN_TRAIN, CONLL2012_EN_DEV, CONLL2012_EN_TEST], outputs):
        in_path = get_resource(in_path)
        files = sorted(glob.glob(f'{in_path}/**/*gold_conll', recursive=True))
        merge_files(files, out_path)
    return outputs
