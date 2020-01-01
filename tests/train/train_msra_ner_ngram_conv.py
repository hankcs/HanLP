# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 23:15
from hanlp.components.ner import RNNNamedEntityRecognizer, NgramConvNamedEntityRecognizer
from hanlp.datasets.ner.msra import MSRA_NER_TRAIN, MSRA_NER_VALID, MSRA_NER_TEST
from hanlp.pretrained.word2vec import RADICAL_CHAR_EMBEDDING_100, CONVSEG_W2V_NEWS_TENSITE_CHAR, \
    CONVSEG_W2V_NEWS_TENSITE_WORD_MSR
from tests import cdroot

cdroot()
recognizer = NgramConvNamedEntityRecognizer()
save_dir = 'data/model/ner/msra_ner_ngram_conv'
recognizer.fit(MSRA_NER_TRAIN, MSRA_NER_VALID, save_dir,
               word_embed={'class_name': 'HanLP>Word2VecEmbedding',
                           'config': {
                               'trainable': True,
                               'filepath': CONVSEG_W2V_NEWS_TENSITE_CHAR,
                               'expand_vocab': False,
                               'lowercase': False,
                           }},
               ngram_embed={'class_name': 'HanLP>Word2VecEmbedding',
                            'config': {
                                'trainable': True,
                                'filepath': CONVSEG_W2V_NEWS_TENSITE_WORD_MSR,
                                'expand_vocab': True,
                                'lowercase': False,
                            }},
               weight_norm=True)
recognizer.evaluate(MSRA_NER_TEST, save_dir)
