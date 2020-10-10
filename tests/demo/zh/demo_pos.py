# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 21:25
import hanlp
from hanlp.pretrained.pos import CTB9_POS_ALBERT_BASE
tagger = hanlp.load(CTB9_POS_ALBERT_BASE)
print(tagger.predict(['我', '的', '希望', '是', '希望', '和平']))
print(tagger.predict([['支持', '批处理'], ['速度', '更', '快']]))
