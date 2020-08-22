# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 22:22

from hanlp.components.tok import TransformerTokenizer
from hanlp.datasets.cws.ctb import CTB6_CWS_TRAIN, CTB6_CWS_VALID, CTB6_CWS_TEST
from hanlp.datasets.cws.sighan2005.msr import SIGHAN2005_MSR_VALID, SIGHAN2005_MSR_TRAIN
from tests import cdroot

cdroot()
tokenizer = TransformerTokenizer()
save_dir = 'data/model/msr_cws_albert_base'
tokenizer.fit(SIGHAN2005_MSR_TRAIN, SIGHAN2005_MSR_VALID, save_dir,
              transformer='albert_base_zh',
              max_seq_length=150,
              metrics='f1', learning_rate=5e-5, epochs=10)
tokenizer.load(save_dir)
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
tokenizer.evaluate(CTB6_CWS_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
