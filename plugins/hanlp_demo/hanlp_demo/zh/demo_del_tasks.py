# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-02-03 13:28
import hanlp
from hanlp.components.mtl.multi_task_learning import MultiTaskLearning
from hanlp_common.document import Document

HanLP: MultiTaskLearning = hanlp.load(hanlp.pretrained.mtl.OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
tasks = list(HanLP.tasks.keys())
print(tasks)  # Pick what you need from what we have
for task in tasks:
    if task not in ('tok', 'pos'):
        del HanLP[task]
# You can save it as a new component
# HanLP.save('path/to/new/component')
# HanLP.load('path/to/new/component')
print(HanLP.tasks.keys())
doc: Document = HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', 'up主来到北京立方庭参观自然语义科技公司。'])
print(doc)
doc.pretty_print()
# Specify which annotation to use
# doc.pretty_print(ner='ner/ontonotes', pos='pku')
