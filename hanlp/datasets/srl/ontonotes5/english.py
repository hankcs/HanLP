# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-25 18:48


from urllib.error import HTTPError

from hanlp.datasets.srl.ontonotes5 import ONTONOTES5_HOME, CONLL12_HOME
from hanlp.datasets.srl.ontonotes5._utils import make_gold_conll, make_ontonotes_language_jsonlines, \
    batch_make_ner_tsv_if_necessary, batch_make_pos_tsv_if_necessary, batch_make_con_txt_if_necessary, \
    batch_make_dep_conllx_if_necessary
from hanlp.utils.io_util import get_resource, path_from_url
from hanlp.utils.log_util import cprint

_ONTONOTES5_ENGLISH_HOME = ONTONOTES5_HOME + 'files/data/english/'
_ONTONOTES5_CONLL12_ENGLISH_HOME = CONLL12_HOME + 'english/'

ONTONOTES5_CONLL12_ENGLISH_TRAIN = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'train.english.conll12.jsonlines'
'''Training set of English OntoNotes5 used in CoNLL12 (:cite:`pradhan-etal-2012-conll`).'''
ONTONOTES5_CONLL12_ENGLISH_DEV = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'development.english.conll12.jsonlines'
'''Dev set of English OntoNotes5 used in CoNLL12 (:cite:`pradhan-etal-2012-conll`).'''
ONTONOTES5_CONLL12_ENGLISH_TEST = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'test.english.conll12.jsonlines'
'''Test set of English OntoNotes5 used in CoNLL12 (:cite:`pradhan-etal-2012-conll`).'''

ONTONOTES5_ENGLISH_TRAIN = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'train.english.v4.jsonlines'
ONTONOTES5_ENGLISH_DEV = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'development.english.v4.jsonlines'
ONTONOTES5_ENGLISH_TEST = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'test.english.v4.jsonlines'

ONTONOTES5_CONLL_ENGLISH_TRAIN = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'train.english.v4_gold_conll'
ONTONOTES5_CONLL_ENGLISH_DEV = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'development.english.v4_gold_conll'
ONTONOTES5_CONLL_ENGLISH_TEST = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'test.english.v4_gold_conll'

ONTONOTES5_POS_ENGLISH_TRAIN = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'train.english.v4.pos.tsv'
ONTONOTES5_POS_ENGLISH_DEV = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'development.english.v4.pos.tsv'
ONTONOTES5_POS_ENGLISH_TEST = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'test.english.v4.pos.tsv'

ONTONOTES5_CON_ENGLISH_TRAIN = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'train.english.con.txt'
ONTONOTES5_CON_ENGLISH_DEV = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'development.english.con.txt'
ONTONOTES5_CON_ENGLISH_TEST = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'test.english.con.txt'

ONTONOTES5_DEP_ENGLISH_TRAIN = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'train.english.dep.conllx'
ONTONOTES5_DEP_ENGLISH_DEV = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'development.english.dep.conllx'
ONTONOTES5_DEP_ENGLISH_TEST = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'test.english.dep.conllx'

# ONTONOTES5_CON_ENGLISH_NOEC_TRAIN = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'train.english.con.noempty.txt'
# ONTONOTES5_CON_ENGLISH_NOEC_DEV = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'development.english.con.noempty.txt'
# ONTONOTES5_CON_ENGLISH_NOEC_TEST = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'test.english.con.noempty.txt'

ONTONOTES5_CONLL12_NER_ENGLISH_TRAIN = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'train.english.conll12.ner.tsv'
'''Training set of English OntoNotes5 used in CoNLL12 (:cite:`pradhan-etal-2012-conll`).'''
ONTONOTES5_CONLL12_NER_ENGLISH_DEV = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'development.english.conll12.ner.tsv'
'''Dev set of English OntoNotes5 used in CoNLL12 (:cite:`pradhan-etal-2012-conll`).'''
ONTONOTES5_CONLL12_NER_ENGLISH_TEST = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'test.english.conll12.ner.tsv'
'''Test set of English OntoNotes5 used in CoNLL12 (:cite:`pradhan-etal-2012-conll`).'''

ONTONOTES5_NER_ENGLISH_TRAIN = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'train.english.v4.ner.tsv'
ONTONOTES5_NER_ENGLISH_DEV = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'development.english.v4.ner.tsv'
ONTONOTES5_NER_ENGLISH_TEST = _ONTONOTES5_CONLL12_ENGLISH_HOME + 'test.english.v4.ner.tsv'

try:
    get_resource(ONTONOTES5_HOME, verbose=False)
except HTTPError:
    intended_file_path = path_from_url(ONTONOTES5_HOME)
    cprint('Ontonotes 5.0 is a [red][bold]copyright[/bold][/red] dataset owned by LDC which we cannot re-distribute. '
           f'Please apply for a licence from LDC (https://catalog.ldc.upenn.edu/LDC2016T13) '
           f'then download it to {intended_file_path}')
    exit(1)

try:
    get_resource(ONTONOTES5_CONLL12_ENGLISH_TRAIN, verbose=False)
except HTTPError:
    make_gold_conll(ONTONOTES5_HOME + '..', 'english')
    make_ontonotes_language_jsonlines(CONLL12_HOME + 'v4', language='english')

batch_make_ner_tsv_if_necessary(
    [ONTONOTES5_CONLL12_ENGLISH_TRAIN, ONTONOTES5_CONLL12_ENGLISH_DEV, ONTONOTES5_CONLL12_ENGLISH_TEST])

batch_make_ner_tsv_if_necessary(
    [ONTONOTES5_ENGLISH_TRAIN, ONTONOTES5_ENGLISH_DEV, ONTONOTES5_ENGLISH_TEST])

batch_make_pos_tsv_if_necessary(
    [ONTONOTES5_ENGLISH_TRAIN, ONTONOTES5_ENGLISH_DEV, ONTONOTES5_ENGLISH_TEST])

batch_make_con_txt_if_necessary(
    [ONTONOTES5_CONLL_ENGLISH_TRAIN, ONTONOTES5_CONLL_ENGLISH_DEV, ONTONOTES5_CONLL_ENGLISH_TEST])

batch_make_dep_conllx_if_necessary(
    [ONTONOTES5_CON_ENGLISH_TRAIN, ONTONOTES5_CON_ENGLISH_DEV, ONTONOTES5_CON_ENGLISH_TEST])

# batch_remove_empty_category_if_necessary(
#     [ONTONOTES5_CON_ENGLISH_TRAIN, ONTONOTES5_CON_ENGLISH_DEV, ONTONOTES5_CON_ENGLISH_TEST])
