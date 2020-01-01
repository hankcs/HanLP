# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-01 17:55
import hanlp

syntactic_parser = hanlp.load(hanlp.pretrained.dep.PTB_BIAFFINE_DEP)
sent = [('Is', 'VBZ'),
        ('this', 'DT'),
        ('the', 'DT'),
        ('future', 'NN'),
        ('of', 'IN'),
        ('chamber', 'NN'),
        ('music', 'NN'),
        ('?', '.')]
tree = syntactic_parser.predict(sent)
print(tree)
