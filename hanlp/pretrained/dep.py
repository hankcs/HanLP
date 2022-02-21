# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 02:55
from hanlp_common.constant import HANLP_URL

CTB5_BIAFFINE_DEP_ZH = HANLP_URL + 'dep/biaffine_ctb5_20191229_025833.zip'
'Biaffine LSTM model (:cite:`dozat:17a`) trained on CTB5.'
CTB7_BIAFFINE_DEP_ZH = HANLP_URL + 'dep/biaffine_ctb7_20200109_022431.zip'
'Biaffine LSTM model (:cite:`dozat:17a`) trained on CTB7.'
CTB9_DEP_ELECTRA_SMALL = HANLP_URL + 'dep/ctb9_dep_electra_small_20220216_100306.zip'
'Electra small encoder (:cite:`clark2020electra`) with Biaffine decoder (:cite:`dozat:17a`) trained on CTB9-SD330. ' \
'Performance is UAS=87.68% LAS=83.54%.'
PMT1_DEP_ELECTRA_SMALL = HANLP_URL + 'dep/pmt_dep_electra_small_20220218_134518.zip'
'Electra small encoder (:cite:`clark2020electra`) with Biaffine decoder (:cite:`dozat:17a`) trained on PKU ' \
'Multi-view Chinese Treebank (PMT) 1.0 (:cite:`qiu-etal-2014-multi`). Performance is UAS=91.21% LAS=88.65%.'
CTB9_UDC_ELECTRA_SMALL = HANLP_URL + 'dep/udc_dep_electra_small_20220218_095452.zip'
'Electra small encoder (:cite:`clark2020electra`) with Biaffine decoder (:cite:`dozat:17a`) trained on CTB9-UD420. ' \
'Performance is UAS=85.92% LAS=81.13% .'

PTB_BIAFFINE_DEP_EN = HANLP_URL + 'dep/ptb_dep_biaffine_20200101_174624.zip'
'Biaffine LSTM model (:cite:`dozat:17a`) trained on PTB.'

ALL = {}
