# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-03 15:26
import hanlp
from hanlp.components.parsers.conll import CoNLLSentence

# semeval15 offers three independent annotations over the Penn Treebank (PTB)
semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL15_PAS_BIAFFINE_EN)
# semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL15_DM_BIAFFINE_EN)
# semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL15_PSD_BIAFFINE_EN)
sent = [('Is', 'VBZ'),
        ('this', 'DT'),
        ('the', 'DT'),
        ('future', 'NN'),
        ('of', 'IN'),
        ('chamber', 'NN'),
        ('music', 'NN'),
        ('?', '.')]
tree = semantic_parser(sent)  # type:CoNLLSentence
print(tree)
