# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 15:39
from hanlp.components.tokenizers.tok_tf import TransformerTokenizerTF
from hanlp.datasets.tokenization.ctb6 import CTB6_CWS_DEV, CTB6_CWS_TEST
from tests import cdroot

cdroot()
tokenizer = TransformerTokenizerTF()
save_dir = 'data/model/cws_bert_base_100million'
tokenizer.fit('data/cws/large/all.txt', CTB6_CWS_DEV, save_dir, transformer='bert-base-chinese',
              metrics='accuracy', batch_size=32)
tokenizer.load(save_dir, metrics='f1')
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
tokenizer.evaluate(CTB6_CWS_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
