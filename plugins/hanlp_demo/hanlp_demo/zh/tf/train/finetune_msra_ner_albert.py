# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 23:15
import hanlp
from hanlp.components.ner.ner_tf import TransformerNamedEntityRecognizerTF
from hanlp.datasets.ner.msra import MSRA_NER_CHAR_LEVEL_TRAIN, MSRA_NER_CHAR_LEVEL_DEV, MSRA_NER_CHAR_LEVEL_TEST
from tests import cdroot

cdroot()
recognizer = TransformerNamedEntityRecognizerTF()
save_dir = 'data/model/ner/finetune_ner_albert_base_zh_msra'
recognizer.fit(MSRA_NER_CHAR_LEVEL_TRAIN, MSRA_NER_CHAR_LEVEL_DEV, save_dir, transformer='albert_base_zh',
               finetune=hanlp.pretrained.ner.MSRA_NER_ALBERT_BASE_ZH)
recognizer.load(save_dir)
print(recognizer.predict(list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。')))
recognizer.evaluate(MSRA_NER_CHAR_LEVEL_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
