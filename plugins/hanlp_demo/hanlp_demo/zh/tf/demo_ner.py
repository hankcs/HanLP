# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-30 19:52
import hanlp

recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
print(recognizer.predict([list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。'),
                          list('萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。')]))
