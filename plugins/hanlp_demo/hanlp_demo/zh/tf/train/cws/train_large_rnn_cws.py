# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:39
import tensorflow as tf

from hanlp.components.tokenizers.tok_tf import RNNTokenizerTF
from hanlp.datasets.cws.ctb import CTB6_CWS_TEST, CTB6_CWS_DEV
from hanlp.pretrained.word2vec import RADICAL_CHAR_EMBEDDING_100
from tests import cdroot

cdroot()

tokenizer = RNNTokenizerTF()
save_dir = 'data/model/cws/large_rnn_cws'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     epsilon=1e-8, clipnorm=5)
tokenizer.fit('data/cws/large/all.txt',
              CTB6_CWS_DEV,
              save_dir,
              embeddings={'class_name': 'HanLP>Word2VecEmbedding',
                          'config': {
                              'trainable': True,
                              'filepath': RADICAL_CHAR_EMBEDDING_100,
                              'expand_vocab': False,
                              'lowercase': False,
                          }},
              early_stopping_patience=5,
              batch_size=64,
              max_seq_len=64,
              metrics='accuracy'
              )
tokenizer.load(save_dir, metrics='f1')
tokenizer.evaluate(CTB6_CWS_TEST, save_dir=save_dir, output=False)
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
print(f'Model saved in {save_dir}')
