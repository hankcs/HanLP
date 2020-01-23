# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-01 03:52
import hanlp

classifier = hanlp.load('SST2_ALBERT_BASE_EN')
print(classifier.predict('I feel lucky'))
