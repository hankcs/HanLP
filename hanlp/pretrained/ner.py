# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-30 20:07
from hanlp_common.constant import HANLP_URL

MSRA_NER_BERT_BASE_ZH = HANLP_URL + 'ner/ner_bert_base_msra_20211227_114712.zip'
'BERT model (:cite:`devlin-etal-2019-bert`) trained on MSRA with 3 entity types.'
MSRA_NER_ALBERT_BASE_ZH = HANLP_URL + 'ner/msra_ner_albert_base_20211228_173323.zip'
'ALBERT model (:cite:`Lan2020ALBERT:`) trained on MSRA with 3 entity types.'
MSRA_NER_ELECTRA_SMALL_ZH = HANLP_URL + 'ner/msra_ner_electra_small_20220215_205503.zip'
'Electra small model (:cite:`clark2020electra`) trained on MSRA with 26 entity types. F1 = `95.16`'
CONLL03_NER_BERT_BASE_CASED_EN = HANLP_URL + 'ner/ner_conll03_bert_base_cased_en_20211227_121443.zip'
'BERT model (:cite:`devlin-etal-2019-bert`) trained on CoNLL03.'

ALL = {}
