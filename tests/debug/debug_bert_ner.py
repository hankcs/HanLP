# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-09 00:06
import hanlp

recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
print(recognizer([list('孽债 （上海话）')]))
print(recognizer(['超', '长'] * 256))
