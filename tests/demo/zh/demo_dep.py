# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 21:25
import hanlp

syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
sent = [('中国', 'NR'),('批准', 'VV'),('设立', 'VV'),('了', 'AS'),('三十万', 'CD'),('家', 'M'),('外商', 'NN'),('投资', 'NN'), ('企业', 'NN')]
tree = syntactic_parser.predict(sent)
print(tree)
