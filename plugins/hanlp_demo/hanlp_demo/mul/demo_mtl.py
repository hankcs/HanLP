# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-31 13:51
import hanlp
from hanlp_common.document import Document

HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
doc: Document = HanLP([
    'In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environment.',
    '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
    '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。',
])
print(doc)
doc.pretty_print()
