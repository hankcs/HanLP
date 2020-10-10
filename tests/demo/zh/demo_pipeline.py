# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 03:24

import hanlp

tokenizer = hanlp.load('LARGE_ALBERT_BASE')
tagger = hanlp.load('CTB5_POS_RNN')
syntactic_parser = hanlp.load('CTB7_BIAFFINE_DEP_ZH')
semantic_parser = hanlp.load('SEMEVAL16_TEXT_BIAFFINE_ZH')

pipeline = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(tokenizer, output_key='tokens') \
    .append(tagger, output_key='part_of_speech_tags') \
    .append(syntactic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='syntactic_dependencies', conll=False) \
    .append(semantic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='semantic_dependencies', conll=False)
print(pipeline)

text = '''HanLP是一系列模型与算法组成的自然语言处理工具包，目标是普及自然语言处理在生产环境中的应用。
HanLP具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点。
内部算法经过工业界和学术界考验，配套书籍《自然语言处理入门》已经出版。
'''

doc = pipeline(text)
print(doc)
# By default the doc is json serializable, it holds true if your pipes output json serializable object too.
# print(json.dumps(doc, ensure_ascii=False, indent=2))

# You can save the config to disk for deploying or sharing.
pipeline.save('zh.json')
# Then load it smoothly.
deployed = hanlp.load('zh.json')
print(deployed)
print(deployed(text))
