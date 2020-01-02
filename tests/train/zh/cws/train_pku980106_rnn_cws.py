# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:39
import tensorflow as tf

from hanlp.components.tok import RNNTokenizer
from hanlp.pretrained.word2vec import RADICAL_CHAR_EMBEDDING_100
from tests import cdroot

cdroot()

tokenizer = RNNTokenizer()
save_dir = 'data/model/cws/pku_6m_rnn_cws'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     epsilon=1e-8, clipnorm=5)
tokenizer.fit('data/cws/pku98/199801-06-seg.txt',
              'data/cws/pku98/pku98_test.txt',
              save_dir,
              embeddings={'class_name': 'HanLP>Word2VecEmbedding',
                          'config': {
                              'trainable': False,
                              'filepath': RADICAL_CHAR_EMBEDDING_100,
                              'expand_vocab': True,
                              'lowercase': False,
                          }}
              )
tokenizer.evaluate('data/cws/pku98/pku98_test.txt', save_dir=save_dir, output=False)
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
print(f'Model saved in {save_dir}')
