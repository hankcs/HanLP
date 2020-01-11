# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-10 21:27
import hanlp

syntactic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL16_NEWS_BIAFFINE_ZH)
syntactic_parser.evaluate(hanlp.datasets.parsing.semeval2016.SEMEVAL2016_NEWS_TEST)
