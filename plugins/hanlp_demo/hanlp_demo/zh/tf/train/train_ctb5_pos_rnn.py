# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 22:46
from hanlp.components.pos_tf import RNNPartOfSpeechTaggerTF
from hanlp.datasets.parsing.ctb5 import CIP_W2V_100_CN
from hanlp.datasets.pos.ctb5 import CTB5_POS_TRAIN, CTB5_POS_DEV, CTB5_POS_TEST
from hanlp.pretrained.fasttext import FASTTEXT_CC_300_EN, FASTTEXT_WIKI_300_ZH
from tests import cdroot

cdroot()
tagger = RNNPartOfSpeechTaggerTF()
save_dir = 'data/model/pos/ctb5_pos_rnn_fasttext'
tagger.fit(CTB5_POS_TRAIN, CTB5_POS_DEV, save_dir, embeddings={'class_name': 'HanLP>FastTextEmbedding',
                                                                 'config': {'filepath': FASTTEXT_WIKI_300_ZH}}, )
tagger.evaluate(CTB5_POS_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
