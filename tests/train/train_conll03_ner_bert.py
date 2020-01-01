# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-25 21:34

from hanlp.components.taggers.transformers.transformer_tagger import TransformerTagger
from hanlp.datasets.ner.conll03 import CONLL03_EN_TRAIN, CONLL03_EN_VALID, CONLL03_EN_TEST
from tests import cdroot

cdroot()
tagger = TransformerTagger()
save_dir = 'data/model/ner-rnn-debug'
tagger.fit(CONLL03_EN_TRAIN, CONLL03_EN_VALID, save_dir, transformer='bert-base-uncased',
           metrics='f1'
           )
tagger.load(save_dir)
# print(tagger.predict('West Indian all-rounder Phil Simmons eats apple .'.split()))
# print(tagger.predict([['This', 'is', 'an', 'old', 'story'],
#                       ['Not', 'this', 'year', '.']]))
# [['DT', 'VBZ', 'DT', 'JJ', 'NN'], ['RB', 'DT', 'NN', '.']]
tagger.evaluate(CONLL03_EN_TEST, save_dir=save_dir, output=False, batch_size=32)
