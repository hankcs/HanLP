# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:39
from hanlp.components.tok import TransformerTokenizer
from hanlp.datasets.cws.sighan2005.msr import SIGHAN2005_MSR_TRAIN, SIGHAN2005_MSR_VALID, SIGHAN2005_MSR_TEST
from tests import cdroot

cdroot()
tokenizer = TransformerTokenizer()
save_dir = 'data/model/cws_bert_base_msra'
tokenizer.fit(SIGHAN2005_MSR_TRAIN, SIGHAN2005_MSR_VALID, save_dir, transformer='chinese_L-12_H-768_A-12',
              metrics='f1')
# tokenizer.load(save_dir)
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
tokenizer.evaluate(SIGHAN2005_MSR_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
