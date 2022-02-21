# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-02 19:41
from hanlp.utils.lang.en.english_tokenizer import tokenize_english

text = """\
Don't go gentle into that good night.
"""
print(tokenize_english(text))
