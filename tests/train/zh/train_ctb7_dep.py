# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 18:33
from hanlp.components.parsers.biaffine_parser import BiaffineDependencyParser
from hanlp.datasets.parsing.ctb import CTB7_DEP_TRAIN, CTB7_DEP_VALID, CTB7_DEP_TEST, CIP_W2V_100_CN
from tests import cdroot

cdroot()
save_dir = 'data/model/dep/biaffine_ctb7'
parser = BiaffineDependencyParser()
parser.fit(CTB7_DEP_TRAIN, CTB7_DEP_VALID, save_dir,
           pretrained_embed={'class_name': 'HanLP>Word2VecEmbedding',
                             'config': {
                                 'trainable': False,
                                 'embeddings_initializer': 'zero',
                                 'filepath': CIP_W2V_100_CN,
                                 'expand_vocab': True,
                                 'lowercase': True,
                                 'normalize': True,
                             }},
           )
parser.load(save_dir)
sentence = [('中国', 'NR'), ('批准', 'VV'), ('设立', 'VV'), ('外商', 'NN'), ('投资', 'NN'), ('企业', 'NN'), ('逾', 'VV'),
            ('三十万', 'CD'), ('家', 'M')]
print(parser.predict(sentence))
parser.evaluate(CTB7_DEP_TEST, save_dir)
