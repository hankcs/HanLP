# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 00:51
from hanlp_common.conll import CoNLLSentence
import os

from hanlp.utils.io_util import get_resource, merge_files
from hanlp_common.io import eprint

_SEMEVAL2016_HOME = 'https://github.com/HIT-SCIR/SemEval-2016/archive/master.zip'

SEMEVAL2016_NEWS_TRAIN = _SEMEVAL2016_HOME + '#train/news.train.conll'
SEMEVAL2016_NEWS_DEV = _SEMEVAL2016_HOME + '#validation/news.valid.conll'
SEMEVAL2016_NEWS_TEST = _SEMEVAL2016_HOME + '#test/news.test.conll'

SEMEVAL2016_NEWS_TRAIN_CONLLU = _SEMEVAL2016_HOME + '#train/news.train.conllu'
SEMEVAL2016_NEWS_DEV_CONLLU = _SEMEVAL2016_HOME + '#validation/news.valid.conllu'
SEMEVAL2016_NEWS_TEST_CONLLU = _SEMEVAL2016_HOME + '#test/news.test.conllu'

SEMEVAL2016_TEXT_TRAIN = _SEMEVAL2016_HOME + '#train/text.train.conll'
SEMEVAL2016_TEXT_DEV = _SEMEVAL2016_HOME + '#validation/text.valid.conll'
SEMEVAL2016_TEXT_TEST = _SEMEVAL2016_HOME + '#test/text.test.conll'

SEMEVAL2016_TEXT_TRAIN_CONLLU = _SEMEVAL2016_HOME + '#train/text.train.conllu'
SEMEVAL2016_TEXT_DEV_CONLLU = _SEMEVAL2016_HOME + '#validation/text.valid.conllu'
SEMEVAL2016_TEXT_TEST_CONLLU = _SEMEVAL2016_HOME + '#test/text.test.conllu'

SEMEVAL2016_FULL_TRAIN_CONLLU = _SEMEVAL2016_HOME + '#train/full.train.conllu'
SEMEVAL2016_FULL_DEV_CONLLU = _SEMEVAL2016_HOME + '#validation/full.valid.conllu'
SEMEVAL2016_FULL_TEST_CONLLU = _SEMEVAL2016_HOME + '#test/full.test.conllu'


def convert_conll_to_conllu(path):
    sents = CoNLLSentence.from_file(path, conllu=True)
    with open(os.path.splitext(path)[0] + '.conllu', 'w') as out:
        for sent in sents:
            for word in sent:
                if not word.deps:
                    word.deps = [(word.head, word.deprel)]
                    word.head = None
                    word.deprel = None
            out.write(str(sent))
            out.write('\n\n')


for file in [SEMEVAL2016_NEWS_TRAIN, SEMEVAL2016_NEWS_DEV, SEMEVAL2016_NEWS_TEST,
             SEMEVAL2016_TEXT_TRAIN, SEMEVAL2016_TEXT_DEV, SEMEVAL2016_TEXT_TEST]:
    file = get_resource(file)
    conllu = os.path.splitext(file)[0] + '.conllu'
    if not os.path.isfile(conllu):
        eprint(f'Converting {os.path.basename(file)} to {os.path.basename(conllu)} ...')
        convert_conll_to_conllu(file)

for group, part in zip([[SEMEVAL2016_NEWS_TRAIN_CONLLU, SEMEVAL2016_TEXT_TRAIN_CONLLU],
                        [SEMEVAL2016_NEWS_DEV_CONLLU, SEMEVAL2016_TEXT_DEV_CONLLU],
                        [SEMEVAL2016_NEWS_TEST_CONLLU, SEMEVAL2016_TEXT_TEST_CONLLU]],
                       ['train', 'valid', 'test']):
    root = get_resource(_SEMEVAL2016_HOME)
    dst = f'{root}/train/full.{part}.conllu'
    if not os.path.isfile(dst):
        group = [get_resource(x) for x in group]
        eprint(f'Concatenating {os.path.basename(group[0])} and {os.path.basename(group[1])} '
               f'into full dataset {os.path.basename(dst)} ...')
        merge_files(group, dst)
