# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-01 03:52
import hanlp

classifier = hanlp.load('CHNSENTICORP_BERT_BASE')
print(classifier.predict('前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！'))
