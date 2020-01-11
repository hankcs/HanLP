# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-10 21:27
import hanlp

syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
syntactic_parser.evaluate(hanlp.datasets.parsing.ctb.CTB7_DEP_TEST)
