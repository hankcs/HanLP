# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 21:12
from hanlp_common.constant import HANLP_URL

SIGHAN2005_PKU_CONVSEG = HANLP_URL + 'tok/sighan2005-pku-convseg_20200110_153722.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on sighan2005 pku dataset.'
SIGHAN2005_MSR_CONVSEG = HANLP_URL + 'tok/convseg-msr-nocrf-noembed_20200110_153524.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on sighan2005 msr dataset.'
CTB6_CONVSEG = HANLP_URL + 'tok/ctb6_convseg_nowe_nocrf_20200110_004046.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on CTB6 dataset.'
PKU_NAME_MERGED_SIX_MONTHS_CONVSEG = HANLP_URL + 'tok/pku98_6m_conv_ngram_20200110_134736.zip'
'Conv model (:cite:`wang-xu-2017-convolutional`) trained on pku98 six months dataset with familiy name and given name merged into one unit.'
LARGE_ALBERT_BASE = HANLP_URL + 'tok/large_corpus_cws_albert_base_20211228_160926.zip'
'ALBERT model (:cite:`Lan2020ALBERT:`) trained on the largest CWS dataset in the world.'
SIGHAN2005_PKU_BERT_BASE_ZH = HANLP_URL + 'tok/sighan2005_pku_bert_base_zh_20201231_141130.zip'
'BERT model (:cite:`devlin-etal-2019-bert`) trained on sighan2005 pku dataset.'
COARSE_ELECTRA_SMALL_ZH = HANLP_URL + 'tok/coarse_electra_small_20220220_013548.zip'
'Electra (:cite:`clark2020electra`) small model trained on coarse-grained CWS corpora. Its performance is P=96.97% R=96.87% F1=96.92% which is ' \
'much higher than that of MTL model '
FINE_ELECTRA_SMALL_ZH = HANLP_URL + 'tok/fine_electra_small_20220217_190117.zip'
'Electra (:cite:`clark2020electra`) small model trained on fine-grained CWS corpora. Its performance is P=97.44% R=97.40% F1=97.42% which is ' \
'much higher than that of MTL model '
CTB9_TOK_ELECTRA_SMALL = HANLP_URL + 'tok/ctb9_electra_small_20220215_205427.zip'
'Electra (:cite:`clark2020electra`) small model trained on CTB9. Its performance is P=97.15% R=97.36% F1=97.26% which is ' \
'much higher than that of MTL model '
CTB9_TOK_ELECTRA_BASE = 'http://download.hanlp.com/tok/extra/ctb9_tok_electra_base_20220426_111949.zip'
'Electra (:cite:`clark2020electra`) base model trained on CTB9. Its performance is ``P: 97.62% R: 97.67% F1: 97.65%`` ' \
'which is much higher than that of MTL model '
CTB9_TOK_ELECTRA_BASE_CRF = 'http://download.hanlp.com/tok/extra/ctb9_tok_electra_base_crf_20220426_161255.zip'
'Electra (:cite:`clark2020electra`) base model trained on CTB9. Its performance is ``P: 97.68% R: 97.71% F1: 97.69%`` ' \
'which is much higher than that of MTL model '
MSR_TOK_ELECTRA_BASE_CRF = 'http://download.hanlp.com/tok/extra/msra_crf_electra_base_20220507_113936.zip'
'Electra (:cite:`clark2020electra`) base model trained on MSR CWS dataset. Its performance is ``P: 98.71% R: 98.64% F1: 98.68%`` ' \
'which is much higher than that of MTL model '

# Will be filled up during runtime
ALL = {}
