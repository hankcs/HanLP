# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-03 23:00
from hanlp.components.classifiers.bert_text_classifier import BertTextClassifier
from hanlp.datasets.glue import STANFORD_SENTIMENT_TREEBANK_2_TRAIN, STANFORD_SENTIMENT_TREEBANK_2_TEST, \
    STANFORD_SENTIMENT_TREEBANK_2_VALID

save_dir = 'data/model/classification/sst2_bert_base_uncased_en'
classifier = BertTextClassifier()
classifier.fit(STANFORD_SENTIMENT_TREEBANK_2_TRAIN, STANFORD_SENTIMENT_TREEBANK_2_VALID, save_dir,
               bert='bert-base-uncased')
classifier.load(save_dir)
print(classifier.classify('it\' s a charming and often affecting journey'))
classifier.evaluate(STANFORD_SENTIMENT_TREEBANK_2_TEST, save_dir=save_dir)
