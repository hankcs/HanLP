# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-12-28 20:47
import hanlp

# Pipeline allows blending multiple callable functions no matter they are a rule, a TensorFlow component or a PyTorch
# one. However, it's slower than the MTL framework.
# pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ALBERT_BASE)  # In case both tf and torch are used, load tf first.

HanLP = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(hanlp.load('CTB9_TOK_ELECTRA_SMALL'), output_key='tok') \
    .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
    .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
    .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=False), output_key='dep', input_key='tok') \
    .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')

doc = HanLP('2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。')
print(doc)
doc.pretty_print()
