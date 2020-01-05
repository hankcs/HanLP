# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-02 19:41
import hanlp

text = """\
Mr. Hankcs bought hankcs.com for 1.5 thousand dollars, i.e. he paid a lot for it. 
Did he mind? Hankcs He Jr. thinks he didn't. In any case, this isn't true... 
Well, with a probability of .9 it isn't or they arent.
"""
print(hanlp.utils.rules.tokenize_english(text))
