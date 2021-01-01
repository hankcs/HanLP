# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 21:25
import hanlp

syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
sent = [('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]
tree = syntactic_parser(sent)
print(tree)
