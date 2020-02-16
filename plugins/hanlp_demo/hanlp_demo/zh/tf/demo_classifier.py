# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-01 03:52
from hanlp.datasets.classification.sentiment import CHNSENTICORP_ERNIE_TEST

import hanlp

classifier = hanlp.load('CHNSENTICORP_BERT_BASE_ZH')
print(classifier.predict('前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！'))

# predict a whole file in batch mode
outputs = classifier.predict(classifier.transform.file_to_inputs(CHNSENTICORP_ERNIE_TEST), gold=True)
print(outputs[:5])
