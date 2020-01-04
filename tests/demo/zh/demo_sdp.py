# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 23:55
import hanlp

semantic_parser = hanlp.load('SEMEVAL16_NEWS_BIAFFINE_ZH')
sent = [('中国', 'NR'),
        ('批准', 'VV'),
        ('设立', 'VV'),
        ('了', 'AS'),
        ('三十万', 'CD'),
        ('家', 'M'),
        ('外商', 'NN'),
        ('投资', 'NN'),
        ('企业', 'NN')]
print(semantic_parser.predict(sent))
