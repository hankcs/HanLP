# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-01 03:51
from hanlp_common.constant import HANLP_URL

CHNSENTICORP_BERT_BASE_ZH = HANLP_URL + 'classification/chnsenticorp_bert_base_20211228_163210.zip'
SST2_ALBERT_BASE_EN = HANLP_URL + 'classification/sst2_albert_base_20211228_164917.zip'

LID_176_FASTTEXT_BASE = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
'''
126MB FastText model for language identification trained on data from Wikipedia, Tatoeba and SETimes.
'''
LID_176_FASTTEXT_SMALL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'
'''
917kB FastText model for language identification trained on data from Wikipedia, Tatoeba and SETimes.
'''

ALL = {}
