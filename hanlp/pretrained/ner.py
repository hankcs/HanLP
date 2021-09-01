# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-30 20:07
from hanlp_common.constant import HANLP_URL

MSRA_NER_BERT_BASE_ZH = HANLP_URL + 'ner/ner_bert_base_msra_20200104_185735.zip'
'BERT model (:cite:`devlin-etal-2019-bert`) trained on MSRA with 3 entity types.'
MSRA_NER_ALBERT_BASE_ZH = HANLP_URL + 'ner/ner_albert_base_zh_msra_20200111_202919.zip'
'ALBERT model (:cite:`Lan2020ALBERT:`) trained on MSRA with 3 entity types.'
MSRA_NER_ELECTRA_SMALL_ZH = HANLP_URL + 'ner/msra_ner_electra_small_20210807_154832.zip'
'Electra small model (:cite:`clark2020electra`) trained on MSRA with 26 entity types. F1 = `95.10`'
CONLL03_NER_BERT_BASE_UNCASED_EN = HANLP_URL + 'ner/ner_conll03_bert_base_uncased_en_20200104_194352.zip'
'BERT model (:cite:`devlin-etal-2019-bert`) trained on CoNLL03.'

ALL = {}
