# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-15 22:26
import hanlp
from hanlp.components.mtl.multi_task_learning import MultiTaskLearning
from hanlp.components.mtl.tasks.pos import TransformerTagging
from hanlp.components.mtl.tasks.tok.tag_tok import TaggingTokenization
from tests import cdroot

cdroot()
HanLP: MultiTaskLearning = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

# Demonstrates custom dict in part-of-speech tagging
pos: TransformerTagging = HanLP['pos/ctb']
pos.dict_tags = {'HanLP': 'state-of-the-art-tool'}
print(f'自定义单个词性:')
HanLP("HanLP为生产环境带来次世代最先进的多语种NLP技术。", tasks='pos/ctb').pretty_print()
print(f'根据上下文自定义词性:')
pos.dict_tags = {('的', '希望'): ('补语成分', '名词'), '希望': '动词'}
HanLP("我的希望是希望张晚霞的背影被晚霞映红。", tasks='pos/ctb').pretty_print()

# 需要算法基础才能理解，初学者可参考 http://nlp.hankcs.com/book.php
# See also https://hanlp.hankcs.com/docs/api/hanlp/components/taggers/transformer_tagger.html
