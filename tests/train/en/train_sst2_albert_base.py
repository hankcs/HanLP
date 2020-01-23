# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-11-10 17:41
import os

from hanlp.components.classifiers.transformer_classifier import TransformerClassifier

from tests import cdroot

from hanlp.datasets.glue import STANFORD_SENTIMENT_TREEBANK_2_VALID, STANFORD_SENTIMENT_TREEBANK_2_TRAIN, \
    STANFORD_SENTIMENT_TREEBANK_2_TEST

cdroot()
save_dir = os.path.join('data', 'model', 'sst', 'sst2_albert_base')
classifier = TransformerClassifier()
classifier.fit(STANFORD_SENTIMENT_TREEBANK_2_TRAIN, STANFORD_SENTIMENT_TREEBANK_2_VALID, save_dir,
               transformer='albert_base')
classifier.load(save_dir)
print(classifier('it\' s a charming and often affecting journey'))
classifier.evaluate(STANFORD_SENTIMENT_TREEBANK_2_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
