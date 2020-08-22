# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 22:22

from hanlp.components.tok import TransformerTokenizer
from hanlp.datasets.cws.ctb import CTB6_CWS_TRAIN, CTB6_CWS_VALID, CTB6_CWS_TEST
from tests import cdroot

cdroot()
tokenizer = TransformerTokenizer()
save_dir = 'data/model/ctb6_cws_bert_base'
tokenizer.fit(CTB6_CWS_TRAIN, CTB6_CWS_VALID, save_dir, transformer='chinese_L-12_H-768_A-12',
              max_seq_length=150,
              epochs=10,
              metrics='f1')
tokenizer.load(save_dir)
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
tokenizer.evaluate(CTB6_CWS_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
