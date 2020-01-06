# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-26 23:20
from hanlp.pretrained.glove import GLOVE_6B_100D
from hanlp.components.parsers.biaffine_parser import BiaffineDependencyParser, BiaffineSemanticDependencyParser
from tests import cdroot

cdroot()
save_dir = 'data/model/semeval15_biaffine_dm'
parser = BiaffineSemanticDependencyParser()
# parser.fit('data/semeval15/en.dm.train.conll', 'data/semeval15/en.dm.dev.conll', save_dir,
#            pretrained_embed={'class_name': 'HanLP>Word2VecEmbedding',
#                              'config': {
#                                  'trainable': False,
#                                  'embeddings_initializer': 'zero',
#                                  'filepath': GLOVE_6B_100D,
#                                  'expand_vocab': True,
#                                  'lowercase': False,
#                                  'unk': 'unk',
#                                  'normalize': True,
#                                  'name': 'glove.6B.100d'
#                              }},
#            # lstm_dropout=0,
#            # mlp_dropout=0,
#            # embed_dropout=0,
#            epochs=1
#            )
parser.load(save_dir)
parser.save_meta(save_dir)
parser.transform.summarize_vocabs()
sentence = [('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'),
            ('music', 'NN'), ('?', '.')]
print(parser.predict(sentence))
parser.evaluate('data/semeval15/en.id.dm.conll', save_dir)
parser.evaluate('data/semeval15/en.ood.dm.conll', save_dir)
