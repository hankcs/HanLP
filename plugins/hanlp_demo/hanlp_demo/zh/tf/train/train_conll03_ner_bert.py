# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-25 21:34
from hanlp.components.ner.ner_tf import TransformerNamedEntityRecognizerTF
from hanlp.datasets.ner.conll03 import CONLL03_EN_TRAIN, CONLL03_EN_DEV, CONLL03_EN_TEST
from tests import cdroot

cdroot()
tagger = TransformerNamedEntityRecognizerTF()
save_dir = 'data/model/ner/ner_conll03_bert_base_cased_en'
tagger.fit(CONLL03_EN_TRAIN, CONLL03_EN_DEV, save_dir, transformer='bert-base-cased',
           metrics='accuracy')
tagger.load(save_dir, metrics='f1')
print(tagger.predict('West Indian all-rounder Phil Simmons eats apple .'.split()))
tagger.evaluate(CONLL03_EN_TEST, save_dir=save_dir, output=False, batch_size=32)
print(f'Model saved in {save_dir}')
