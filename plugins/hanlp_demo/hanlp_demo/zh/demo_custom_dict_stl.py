# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-15 22:26
import hanlp
from hanlp.components.tokenizers.transformer import TransformerTaggingTokenizer

# 加载单任务模型：
tok: TransformerTaggingTokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

tok.dict_force = tok.dict_combine = None
print(f'不挂词典:\n{tok("首相和川普通电话")}')

tok.dict_force = {'川普'}
print(f'强制模式:\n{tok(["首相和川普通电话", "银川普通人与川普通电话讲四川普通话"])}')  # 慎用，详见《自然语言处理入门》第二章

tok.dict_force = {'川普通电话': ['川普', '通', '电话']}
print(f'强制校正:\n{tok(["首相和川普通电话", "银川普通人与川普通电话讲四川普通话"])}')

tok.dict_force = None
tok.dict_combine = {'美国总统'}
print(f'合并模式:\n{tok("首相和川普通电话，川普是美国总统。")}')

# 需要算法基础才能理解，初学者可参考 http://nlp.hankcs.com/book.php
# See also https://hanlp.hankcs.com/docs/api/hanlp/components/tokenizers/transformer.html
