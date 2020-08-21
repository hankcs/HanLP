# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-25 21:34

import tensorflow as tf

from hanlp.components.ner import RNNNamedEntityRecognizer
from hanlp.datasets.ner.conll03 import CONLL03_EN_TRAIN, CONLL03_EN_TEST
from hanlp.pretrained.glove import GLOVE_6B_100D
from hanlp.pretrained.rnnlm import FLAIR_LM_FW_WMT11_EN, FLAIR_LM_BW_WMT11_EN
from tests import cdroot

cdroot()
tagger = RNNNamedEntityRecognizer()
save_dir = 'data/model/conll03-ner-rnn-flair'
tagger.fit(CONLL03_EN_TRAIN, CONLL03_EN_TEST, save_dir, epochs=100,
           optimizer=tf.keras.optimizers.Adam(learning_rate=0.1,
                                              beta_1=0.9,
                                              beta_2=0.999,
                                              epsilon=1e-8),
           loss='crf',
           rnn_units=256,
           embeddings=[
               {'class_name': 'HanLP>Word2VecEmbedding',
                'config': {
                    'trainable': False,
                    'embeddings_initializer': 'zero',
                    'filepath': GLOVE_6B_100D,
                    'expand_vocab': True,
                    'lowercase': False
                }},
               {'class_name': 'HanLP>ContextualStringEmbedding',
                'config': {
                    'trainable': False,
                    'forward_model_path': FLAIR_LM_FW_WMT11_EN,
                    'backward_model_path': FLAIR_LM_BW_WMT11_EN
                }}
           ],
           rnn_output_dropout=0.5,
           rnn_input_dropout=0.5,
           batch_size=32,
           metrics='f1',
           anneal_factor=0.5,
           patience=2,
           )
print(tagger.predict('West Indian all-rounder Phil Simmons eats apple .'.split()))
# print(tagger.predict([['This', 'is', 'an', 'old', 'story'],
#                       ['Not', 'this', 'year', '.']]))
# [['DT', 'VBZ', 'DT', 'JJ', 'NN'], ['RB', 'DT', 'NN', '.']]
# tagger.load(save_dir)
tagger.evaluate(CONLL03_EN_TEST, save_dir=save_dir, output=False)
