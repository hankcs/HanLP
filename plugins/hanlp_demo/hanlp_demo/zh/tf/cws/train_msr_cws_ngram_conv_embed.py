# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:39
import tensorflow as tf

from hanlp.components.tok import NgramConvTokenizer
from hanlp.datasets.cws.sighan2005.msr import SIGHAN2005_MSR_TRAIN, SIGHAN2005_MSR_VALID, SIGHAN2005_MSR_TEST
from hanlp.pretrained.word2vec import CONVSEG_W2V_NEWS_TENSITE_CHAR, CONVSEG_W2V_NEWS_TENSITE_WORD_MSR
from tests import cdroot

cdroot()
tokenizer = NgramConvTokenizer()
save_dir = 'data/model/cws/convseg-msr-nocrf-noembed'
tokenizer.fit(SIGHAN2005_MSR_TRAIN,
              SIGHAN2005_MSR_VALID,
              save_dir,
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
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                 epsilon=1e-8, clipnorm=5),
              epochs=3,
              window_size=4,
              metrics='f1',
              weight_norm=True)
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
tokenizer.load(save_dir, metrics='f1')
tokenizer.evaluate(SIGHAN2005_MSR_TEST, save_dir=save_dir)
