# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-02-11 09:14
import hanlp

lm = hanlp.load(hanlp.pretrained.rnnlm.FLAIR_LM_FW_WMT11_EN)
print(''.join(lm.generate_text(list('hello'))))
