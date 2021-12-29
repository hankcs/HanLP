# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 22:22
from hanlp.components.tokenizers.tok_tf import TransformerTokenizerTF
from hanlp.datasets.tokenization.ctb6 import CTB6_CWS_DEV, CTB6_CWS_TEST
from tests import cdroot

cdroot()
tokenizer = TransformerTokenizerTF()
save_dir = 'data/model/large_corpus_cws_electra_small'
tokenizer.fit('data/cws/large/all.txt',
              CTB6_CWS_DEV, save_dir,
              transformer='hfl/chinese-electra-small-discriminator',
              max_seq_length=128,
              metrics='accuracy', learning_rate=5e-5, epochs=10)
tokenizer.load(save_dir, metrics='f1')
print(tokenizer.predict(['中央民族乐团离开北京前往维也纳', '商品和服务']))
tokenizer.evaluate(CTB6_CWS_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
