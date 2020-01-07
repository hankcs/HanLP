# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:39
from hanlp.datasets.cws.sighan2005.pku import SIGHAN2005_PKU_TRAIN, SIGHAN2005_PKU_VALID, SIGHAN2005_PKU_TEST
from hanlp.pretrained.word2vec import CONVSEG_W2V_NEWS_TENSITE_CHAR
from hanlp.utils.tf_util import nice
from tests import cdroot
import tensorflow as tf

nice()
cdroot()
from hanlp.components.tok import NgramConvTokenizer

tokenizer = NgramConvTokenizer()
save_dir = 'data/model/cws/sighan2005-pku-convseg'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     epsilon=1e-8, clipnorm=5)
tokenizer.fit(SIGHAN2005_PKU_TRAIN,
              SIGHAN2005_PKU_VALID,
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
tokenizer.evaluate(SIGHAN2005_PKU_TEST, save_dir=save_dir, output=False)
# print(tagger.tag(list('中央民族乐团离开北京前往维也纳')))
# print(tagger.predict('中央民族乐团离开北京前往维也纳'))
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
print(f'Model saved in {save_dir}')
