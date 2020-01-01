# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-26 23:20
from hanlp.datasets.parsing.semeval2016 import SEMEVAL2016_NEWS_TRAIN, SEMEVAL2016_NEWS_VALID, SEMEVAL2016_NEWS_TEST
from hanlp.pretrained.word2vec import SEMEVAL16_EMBEDDINGS_300_NEWS_CN
from hanlp.utils.tf_util import nice

nice()
from hanlp.components.parsers.biaffine_parser import BiaffineSemanticDependencyParser
from tests import cdroot

cdroot()
save_dir = 'data/model/sdp/semeval16-news'
parser = BiaffineSemanticDependencyParser()
parser.fit(SEMEVAL2016_NEWS_TRAIN, SEMEVAL2016_NEWS_VALID, save_dir,
           pretrained_embed={'class_name': 'HanLP>Word2VecEmbedding',
                             'config': {
                                 'trainable': False,
                                 'embeddings_initializer': 'zero',
                                 'filepath': SEMEVAL16_EMBEDDINGS_300_NEWS_CN,
                                 'expand_vocab': True,
                                 'lowercase': True,
                                 'normalize': True,
                             }},
           )
parser.load(save_dir)
sentence = [('中国', 'NR'), ('批准', 'VV'), ('设立', 'VV'), ('外商', 'NN'), ('投资', 'NN'), ('企业', 'NN'), ('逾', 'VV'),
            ('三十万', 'CD'), ('家', 'M')]
print(parser.predict(sentence))
parser.evaluate(SEMEVAL2016_NEWS_TEST, save_dir)
