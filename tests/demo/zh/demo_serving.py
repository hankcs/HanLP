# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-06 20:23
import hanlp
from hanlp.common.component import KerasComponent

tagger: KerasComponent = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN)
print(tagger('商品 和 服务'.split()))
tagger.serve()
