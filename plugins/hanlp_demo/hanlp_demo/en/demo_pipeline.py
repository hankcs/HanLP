# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-04 21:05
import hanlp

tokenizer = hanlp.utils.rules.tokenize_english
tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
syntactic_parser = hanlp.load(hanlp.pretrained.dep.PTB_BIAFFINE_DEP_EN)
semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL15_PAS_BIAFFINE_EN)

pipeline = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(tokenizer, output_key='tokens') \
    .append(tagger, output_key='part_of_speech_tags') \
    .append(syntactic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='syntactic_dependencies', conll=False) \
    .append(semantic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='semantic_dependencies', conll=False)
print(pipeline)

text = '''Jobs and Wozniak co-founded Apple in 1976 to sell Wozniak's Apple I personal computer.
Together the duo gained fame and wealth a year later with the Apple II.
'''

doc = pipeline(text)
print(doc)

# You can save the config to disk for deploying or sharing.
pipeline.save('en.json')
# Then load it smoothly.
deployed = hanlp.load('en.json')
print(deployed)
print(deployed(text))
