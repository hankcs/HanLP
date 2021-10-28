# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-29 11:06
import hanlp
from hanlp.components.mtl.tasks.ner.tag_ner import TaggingNamedEntityRecognition
from hanlp.utils.io_util import get_resource

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH)
ner: TaggingNamedEntityRecognition = HanLP['ner/msra']
ner.dict_whitelist = {'午饭后': 'TIME'}
doc = HanLP('2021年测试高血压是138，时间是午饭后2点45，低血压是44', tasks='ner/msra')
doc.pretty_print()
print(doc['ner/msra'])

ner.dict_tags = {('名字', '叫', '金华'): ('O', 'O', 'S-PERSON')}
HanLP('他在浙江金华出生，他的名字叫金华。', tasks='ner/msra').pretty_print()

# HanLP.save(get_resource(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH))

# 需要算法基础才能理解，初学者可参考 http://nlp.hankcs.com/book.php
# See https://hanlp.hankcs.com/docs/api/hanlp/components/mtl/tasks/ner/tag_ner.html
