# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-17 22:30
import hanlp
from hanlp_common.document import Document

HanLP = hanlp.load(hanlp.pretrained.mtl.NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA)
doc: Document = HanLP([
    '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
    '奈須きのこは1973年11月28日に千葉県円空山で生まれ、ゲーム制作会社「ノーツ」の設立者だ。',
])
print(doc)
doc.pretty_print()
