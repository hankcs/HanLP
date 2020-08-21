# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-25 21:34

import tensorflow as tf

from hanlp.components.pos import RNNPartOfSpeechTagger
from hanlp.pretrained.fasttext import FASTTEXT_CC_300_EN
from tests import cdroot

cdroot()
tagger = RNNPartOfSpeechTagger()
save_dir = 'data/model/pos/ptb_pos_rnn_fasttext'
optimizer = tf.keras.optimizers.SGD(lr=0.015)
# optimizer = 'adam'
tagger.fit('data/ptb-pos/train.tsv',
           'data/ptb-pos/dev.tsv',
           batch_size=10,
           save_dir=save_dir,
           embeddings={'class_name': 'HanLP>FastTextEmbedding',
                       'config': {'filepath': FASTTEXT_CC_300_EN}},
           optimizer=optimizer,
           lr_decay_per_epoch=0.05,
           rnn_units=100,
           rnn_input_dropout=0.5,
           rnn_output_dropout=0.5,
           epochs=100,
           verbose=True)
tagger.load(save_dir)
tagger.evaluate('data/ptb-pos/test.tsv', save_dir=save_dir, output=False)
print(tagger.predict(['This' 'time', 'is', 'for', 'dinner']))
print(tagger.predict([['This', 'is', 'an', 'old', 'story'],
                      ['Not', 'this', 'year', '.']]))
print(f'Model saved in {save_dir}')
