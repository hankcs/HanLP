# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-11 11:08
from hanlp.common.trie import Trie

trie = Trie({'密码', '码'})
print(trie.parse_longest('密码设置'))
