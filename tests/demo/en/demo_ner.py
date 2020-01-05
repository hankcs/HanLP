# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-03 22:50
import hanlp

recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN)
print(recognizer(["President", "Obama", "is", "speaking", "at", "the", "White", "House."]))
