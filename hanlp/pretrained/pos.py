# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 01:57
from hanlp_common.constant import HANLP_URL

CTB5_POS_RNN = HANLP_URL + 'pos/ctb5_pos_rnn_20200113_235925.zip'
'An old school BiLSTM tagging model trained on CTB5.'
CTB5_POS_RNN_FASTTEXT_ZH = HANLP_URL + 'pos/ctb5_pos_rnn_fasttext_20191230_202639.zip'
'An old school BiLSTM tagging model with FastText (:cite:`bojanowski2017enriching`) embeddings trained on CTB5.'
CTB9_POS_ALBERT_BASE = HANLP_URL + 'pos/ctb9_albert_base_20211228_163935.zip'
'ALBERT model (:cite:`Lan2020ALBERT:`) trained on CTB9 (:cite:`https://doi.org/10.35111/gvd0-xk91`). This is a TF component.'
CTB9_POS_ELECTRA_SMALL_TF = HANLP_URL + 'pos/pos_ctb_electra_small_20211227_121341.zip'
'Electra small model (:cite:`clark2020electra`) trained on CTB9 (:cite:`https://doi.org/10.35111/gvd0-xk91`). Accuracy = `96.75`. This is a TF component.'
CTB9_POS_ELECTRA_SMALL = HANLP_URL + 'pos/ctb9_pos_electra_small_20220118_164341.zip'
'Electra small model (:cite:`clark2020electra`) trained on CTB9 (:cite:`https://doi.org/10.35111/gvd0-xk91`). Accuracy = `96.62`.'

C863_POS_ELECTRA_SMALL = HANLP_URL + 'pos/pos_863_electra_small_20210808_124848.zip'
'Electra small model (:cite:`clark2020electra`) trained on Chinese 863 corpus. Accuracy = `95.22`.'

PKU98_POS_ELECTRA_SMALL = HANLP_URL + 'pos/pos_pku_electra_small_20210808_125158.zip'
'Electra small model (:cite:`clark2020electra`) trained on CTB9 (:cite:`https://doi.org/10.35111/gvd0-xk91`). Accuracy = `97.60`.'

PTB_POS_RNN_FASTTEXT_EN = HANLP_URL + 'pos/ptb_pos_rnn_fasttext_20200103_145337.zip'
'An old school BiLSTM tagging model with FastText (:cite:`bojanowski2017enriching`) embeddings trained on PTB.'

ALL = {}
