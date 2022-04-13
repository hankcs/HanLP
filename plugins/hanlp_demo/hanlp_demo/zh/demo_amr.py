# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-04-12 22:19
import hanlp

parser = hanlp.load(hanlp.pretrained.amr.MRP2020_AMR_ENG_ZHO_XLM_BASE)

# For Chinese:
print(parser(["男孩", "希望", "女孩", "相信", "他", "。"]))
print(parser(["男孩", "希望", "女孩", "相信", "他", "。"], output_amr=False))

# For English:
print(parser(['The', 'boy', 'wants', 'the', 'girl', 'to', 'believe', 'him', '.'], language='eng'))
# It's suggested to also feed the lemma for stabler performance.
print(parser([('The', 'the'), ('boy', 'boy'), ('wants', 'want'), ('the', 'the'), ('girl', 'girl'), ('to', 'to'),
              ('believe', 'believe'), ('him', 'he'), ('.', '.')], language='eng'))
