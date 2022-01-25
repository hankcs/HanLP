# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-01-25 19:09
import hanlp

amr_parser = hanlp.load(hanlp.pretrained.amr.AMR3_SEQ2SEQ_BART_LARGE)
amr = amr_parser('The boy wants the girl to believe him.')
print(amr)
