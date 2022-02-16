# -*- coding:utf-8 -*-
# Author=hankcs
# Date=2022-01-18 10:34
from hanlp_common.constant import HANLP_URL

CTB9_CON_ELECTRA_SMALL = HANLP_URL + 'constituency/ctb9_con_electra_small_20220215_230116.zip'
'Electra (:cite:`clark2020electra`) small model trained on CTB9 with major categories. ' \
'Its performance is UCM=39.06% LCM=34.99% UP=90.05% UR=90.01% UF=90.03% LP=87.02% LR=86.98% LF=87.00%.'

CTB9_CON_FULL_TAG_ELECTRA_SMALL = HANLP_URL + 'constituency/ctb9_full_tag_con_electra_small_20220118_103119.zip'
'Electra (:cite:`clark2020electra`) small model trained on CTB9 with full subcategories. ' \
'Its performance is UCM=38.29% LCM=28.95% UP=90.16% UR=90.13% UF=90.15% LP=83.46% LR=83.43% LF=83.45%.'

# Will be filled up during runtime
ALL = {}
