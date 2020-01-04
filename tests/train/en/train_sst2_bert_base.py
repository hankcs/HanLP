# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-03 23:00
from hanlp.components.classifiers.transformer_classifier import TransformerClassifier
from hanlp.datasets.glue import STANFORD_SENTIMENT_TREEBANK_2_TRAIN, STANFORD_SENTIMENT_TREEBANK_2_TEST, \
    STANFORD_SENTIMENT_TREEBANK_2_VALID

save_dir = 'data/model/classification/sst2_bert_base_uncased_en'
classifier = TransformerClassifier()
classifier.fit(STANFORD_SENTIMENT_TREEBANK_2_TRAIN, STANFORD_SENTIMENT_TREEBANK_2_VALID, save_dir,
               transformer='uncased_L-12_H-768_A-12')
classifier.load(save_dir)
print(classifier.predict('it\' s a charming and often affecting journey'))
classifier.evaluate(STANFORD_SENTIMENT_TREEBANK_2_TEST, save_dir=save_dir)
