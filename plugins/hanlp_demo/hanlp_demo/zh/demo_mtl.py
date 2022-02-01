# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-31 13:51
import hanlp
from hanlp_common.document import Document

# CLOSE是自然语义标注的闭源语料库，BASE是中号模型，ZH中文
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
# 默认执行全部任务
doc: Document = HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。'])
# 返回类型Document是dict的子类，打印出来兼容JSON
print(doc)
# 即时可视化，防止换行请最大化窗口，推荐在Jupyter Notebook里调用
doc.pretty_print()
# 指定可视化OntoNotes标准的NER
# doc.pretty_print(ner='ner/ontonotes', pos='pku')
