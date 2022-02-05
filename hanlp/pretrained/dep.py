# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 02:55
from hanlp_common.constant import HANLP_URL

CTB5_BIAFFINE_DEP_ZH = HANLP_URL + 'dep/biaffine_ctb5_20191229_025833.zip'
'Biaffine LSTM model (:cite:`dozat:17a`) trained on CTB5.'
CTB7_BIAFFINE_DEP_ZH = HANLP_URL + 'dep/biaffine_ctb7_20200109_022431.zip'
'Biaffine LSTM model (:cite:`dozat:17a`) trained on CTB7.'
CTB9_BIAFFINE_ELECTARA_SMALL = HANLP_URL + 'dep/ctb9_dep_electra_small_20220204_221541.zip'
'Electra small encoder (:cite:`clark2020electra`) with Biaffine decoder (:cite:`dozat:17a`) trained on CTB9-SD330. ' \
'Performance is UAS=87.74% LAS=83.67%.'

PTB_BIAFFINE_DEP_EN = HANLP_URL + 'dep/ptb_dep_biaffine_20200101_174624.zip'
'Biaffine LSTM model (:cite:`dozat:17a`) trained on PTB.'

ALL = {}
