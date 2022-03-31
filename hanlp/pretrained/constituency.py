# -*- coding:utf-8 -*-
# Author=hankcs
# Date=2022-01-18 10:34
from hanlp_common.constant import HANLP_URL

CTB9_CON_ELECTRA_SMALL = HANLP_URL + 'constituency/ctb9_con_electra_small_20220215_230116.zip'
'Electra (:cite:`clark2020electra`) small tree CRF model (:cite:`ijcai2020-560`) trained on CTB9 with major categories. ' \
'Its performance is UCM=39.06% LCM=34.99% UP=90.05% UR=90.01% UF=90.03% LP=87.02% LR=86.98% LF=87.00%.'

CTB9_CON_FULL_TAG_ELECTRA_SMALL = HANLP_URL + 'constituency/ctb9_full_tag_con_electra_small_20220118_103119.zip'
'Electra (:cite:`clark2020electra`) small tree CRF model (:cite:`ijcai2020-560`) trained on CTB9 with full subcategories. ' \
'Its performance is UCM=38.29% LCM=28.95% UP=90.16% UR=90.13% UF=90.15% LP=83.46% LR=83.43% LF=83.45%.'

CTB9_CON_FULL_TAG_ERNIE_GRAM = 'http://download.hanlp.com/constituency/extra/ctb9_full_tag_con_ernie_20220331_121430.zip'
'ERNIE-GRAM (:cite:`xiao-etal-2021-ernie`) base tree CRF model (:cite:`ijcai2020-560`) trained on CTB9 with full subcategories. ' \
'Its performance is UCM=42.04% LCM=31.72% UP=91.33% UR=91.53% UF=91.43% LP=85.31% LR=85.49% LF=85.40%.'

# Will be filled up during runtime
ALL = {}
