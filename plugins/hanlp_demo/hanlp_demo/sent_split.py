# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-31 14:23
import hanlp

split_sent = hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)
output = split_sent('3.14 is pi. “你好！！！”——他说。劇場版「Fate/stay night [HF]」最終章公開カウントダウン！')
print('\n'.join(output))
# See also https://hanlp.hankcs.com/docs/api/hanlp/components/eos.html
