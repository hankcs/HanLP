# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-02 19:41
import hanlp

text = """\
Mr. Smith bought cheapsite.com for 1.5 million dollars, i.e. he paid a lot for it. Did he mind? Adam Jones Jr. thinks he didn't. In any case, this isn't true... Well, with a probability of .9 it isn't or they arent.
"""
print(hanlp.utils.rules.tokenize_english(text))
