# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-31 13:51
import hanlp
from hanlp_common.document import Document

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
doc: Document = HanLP(['阿 婆主来到北京立方庭参观自然语义科技公司。'], tasks='tok/fine')
print(doc)
doc.pretty_print()
# Specify which annotation to use
# doc.pretty_print(ner='ner/ontonotes', pos='pku')
