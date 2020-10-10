# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 21:25
import hanlp
from hanlp.pretrained.pos import CTB5_POS_RNN
tagger = hanlp.load(CTB5_POS_RNN)
print(tagger.predict(['我', '的', '希望', '是', '希望', '和平']))
print(tagger.predict([['支持', '批处理'], ['速度', '更', '快']]))
