# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-30 21:01
from hanlp.components.classifiers.transformer_classifier import TransformerClassifier, TransformerTextTransform
from hanlp.datasets.classification.sentiment import CHNSENTICORP_ERNIE_TRAIN, CHNSENTICORP_ERNIE_TEST, \
    CHNSENTICORP_ERNIE_VALID
from tests import cdroot

cdroot()
save_dir = 'data/model/classification/chnsenticorp_bert_base'
classifier = TransformerClassifier(TransformerTextTransform(y_column=0))
classifier.fit(CHNSENTICORP_ERNIE_TRAIN, CHNSENTICORP_ERNIE_VALID, save_dir,
               transformer='chinese_L-12_H-768_A-12')
classifier.load(save_dir)
print(classifier.predict('前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！'))
classifier.evaluate(CHNSENTICORP_ERNIE_TEST, save_dir=save_dir)
