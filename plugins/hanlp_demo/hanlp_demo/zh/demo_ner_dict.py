# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-29 11:06
import hanlp

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
HanLP['ner/msra'].dict_whitelist = {'午饭后': 'TIME'}
doc = HanLP('2021年测试高血压是138，时间是午饭后2点45，低血压是44', tasks='ner/msra')
doc.pretty_print()
print(doc['ner/msra'])

# See https://hanlp.hankcs.com/docs/api/hanlp/components/mtl/tasks/ner/tag_ner.html
