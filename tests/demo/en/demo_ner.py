# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-03 22:50
import hanlp

recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_CASED_EN)
print(recognizer.predict(["President", "Obama", "is", "speaking", "at", "the", "White", "House."]))
