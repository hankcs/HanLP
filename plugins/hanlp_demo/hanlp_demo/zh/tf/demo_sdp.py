# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 23:55
import hanlp

semantic_parser = hanlp.load('SEMEVAL16_NEWS_BIAFFINE_ZH')
sent = [('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]
print(semantic_parser(sent))
