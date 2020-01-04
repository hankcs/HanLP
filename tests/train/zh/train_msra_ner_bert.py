# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 23:15
from hanlp.components.ner import TransformerNamedEntityRecognizer
from hanlp.datasets.ner.msra import MSRA_NER_TRAIN, MSRA_NER_VALID, MSRA_NER_TEST
from tests import cdroot

cdroot()
recognizer = TransformerNamedEntityRecognizer()
save_dir = 'data/model/ner/ner_bert_base_msra_2'
recognizer.fit(MSRA_NER_TRAIN, MSRA_NER_VALID, save_dir, transformer='chinese_L-12_H-768_A-12',
               metrics='accuracy')  # accuracy is faster
recognizer.load(save_dir, metrics='f1')
print(recognizer.predict(list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。')))
recognizer.evaluate(MSRA_NER_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
