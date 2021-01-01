# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 23:15
from hanlp.components.ner_tf import RNNNamedEntityRecognizerTF
from hanlp.datasets.ner.msra import MSRA_NER_CHAR_LEVEL_TRAIN, MSRA_NER_CHAR_LEVEL_DEV, MSRA_NER_CHAR_LEVEL_TEST
from hanlp.pretrained.word2vec import RADICAL_CHAR_EMBEDDING_100
from tests import cdroot

cdroot()
recognizer = RNNNamedEntityRecognizerTF()
save_dir = 'data/model/ner/msra_ner_rnn'
recognizer.fit(MSRA_NER_CHAR_LEVEL_TRAIN, MSRA_NER_CHAR_LEVEL_DEV, save_dir,
               embeddings=RADICAL_CHAR_EMBEDDING_100,
               embedding_trainable=True,
               epochs=100)
recognizer.evaluate(MSRA_NER_CHAR_LEVEL_TEST, save_dir)
