# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 21:58

import tensorflow as tf

from hanlp.components.tok import NgramConvTokenizer
from hanlp.datasets.cws.ctb import CTB6_CWS_TRAIN, CTB6_CWS_VALID, CTB6_CWS_TEST
from hanlp.pretrained.word2vec import CONVSEG_W2V_NEWS_TENSITE_CHAR
from tests import cdroot

cdroot()
tokenizer = NgramConvTokenizer()
save_dir = 'data/model/cws/ctb6_cws'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     epsilon=1e-8, clipnorm=5)
tokenizer.fit(CTB6_CWS_TRAIN,
              CTB6_CWS_VALID,
              save_dir,
              word_embed={'class_name': 'HanLP>Word2VecEmbedding',
                          'config': {
                              'trainable': True,
                              'filepath': CONVSEG_W2V_NEWS_TENSITE_CHAR,
                              'expand_vocab': False,
                              'lowercase': False,
                          }},
              optimizer=optimizer,
              window_size=0,
              weight_norm=True)
tokenizer.evaluate(CTB6_CWS_TEST, save_dir=save_dir, output=False)
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
print(f'Model saved in {save_dir}')
