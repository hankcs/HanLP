# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-02 19:41
import hanlp

text = """\
Don't go gentle into that good night.
"""
print(hanlp.utils.rules.tokenize_english(text))
