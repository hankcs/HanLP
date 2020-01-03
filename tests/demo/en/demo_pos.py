# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-03 22:16
import hanlp

tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
print(tagger(['Is', 'this', 'the', 'future', 'of', 'chamber', 'music', '?']))
