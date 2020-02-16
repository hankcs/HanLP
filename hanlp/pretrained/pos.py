# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 01:57
from hanlp_common.constant import HANLP_URL

CTB5_POS_RNN = HANLP_URL + 'pos/ctb5_pos_rnn_20200113_235925.zip'
'An old school BiLSTM tagging model trained on CTB5.'
CTB5_POS_RNN_FASTTEXT_ZH = HANLP_URL + 'pos/ctb5_pos_rnn_fasttext_20191230_202639.zip'
'An old school BiLSTM tagging model with FastText (:cite:`bojanowski2017enriching`) embeddings trained on CTB5.'
CTB9_POS_ALBERT_BASE = HANLP_URL + 'pos/ctb9_albert_base_zh_epoch_20_20201011_090522.zip'
'ALBERT model (:cite:`Lan2020ALBERT:`) trained on CTB9.'

PTB_POS_RNN_FASTTEXT_EN = HANLP_URL + 'pos/ptb_pos_rnn_fasttext_20200103_145337.zip'
'An old school BiLSTM tagging model with FastText (:cite:`bojanowski2017enriching`) embeddings trained on PTB.'

ALL = {}