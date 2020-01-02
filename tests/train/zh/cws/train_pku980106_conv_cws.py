# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:39
import tensorflow as tf

from hanlp.components.tok import NgramConvTokenizer
from hanlp.datasets.cws.sighan2005.pku import SIGHAN2005_PKU_TEST, SIGHAN2005_PKU_VALID
from hanlp.pretrained.word2vec import RADICAL_CHAR_EMBEDDING_100
from tests import cdroot

cdroot()

tokenizer = NgramConvTokenizer()
save_dir = 'data/model/cws/sighan2005-pku-convseg'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     epsilon=1e-8, clipnorm=5)
tokenizer.fit('data/cws/pku98/199801-06-seg.txt',
              SIGHAN2005_PKU_VALID,
              save_dir,
              word_embed={'class_name': 'HanLP>Word2VecEmbedding',
                          'config': {
                              'trainable': False,
                              'filepath': RADICAL_CHAR_EMBEDDING_100,
                              'expand_vocab': True,
                              'lowercase': False,
                          }},
              optimizer=optimizer,
              window_size=0,
              weight_norm=True)
tokenizer.evaluate(SIGHAN2005_PKU_TEST, save_dir=save_dir, output=False)
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
print(f'Model saved in {save_dir}')
