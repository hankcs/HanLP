# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 21:12
from hanlp_common.constant import HANLP_URL

SIGHAN2005_PKU_CONVSEG = HANLP_URL + 'tok/sighan2005-pku-convseg_20200110_153722.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on sighan2005 pku dataset.'
SIGHAN2005_MSR_CONVSEG = HANLP_URL + 'tok/convseg-msr-nocrf-noembed_20200110_153524.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on sighan2005 msr dataset.'
# SIGHAN2005_MSR_BERT_BASE = HANLP_URL + 'tok/cws_bert_base_msra_20191230_194627.zip'
CTB6_CONVSEG = HANLP_URL + 'tok/ctb6_convseg_nowe_nocrf_20200110_004046.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on CTB6 dataset.'
# CTB6_BERT_BASE = HANLP_URL + 'tok/cws_bert_base_ctb6_20191230_185536.zip'
PKU_NAME_MERGED_SIX_MONTHS_CONVSEG = HANLP_URL + 'tok/pku98_6m_conv_ngram_20200110_134736.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on pku98 six months dataset with name merged into one unit.'
LARGE_ALBERT_BASE = HANLP_URL + 'tok/large_cws_albert_base_20200828_011451.zip'
'ALBERT model (:cite:`Lan2020ALBERT:`) trained on the largest CWS dataset in the world.'
SIGHAN2005_PKU_BERT_BASE_ZH = HANLP_URL + 'tok/sighan2005_pku_bert_base_zh_20201231_141130.zip'
'BERT model (:cite:`devlin-etal-2019-bert`) trained on sighan2005 pku dataset.'

# Will be filled up during runtime
ALL = {}
