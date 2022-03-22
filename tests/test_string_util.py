# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-22 17:17
import unittest

from hanlp.utils.string_util import possible_tokenization


class TestStringUtility(unittest.TestCase):
    def test_enumerate_tokenization(self):
        text = '商品和服务'
        toks = possible_tokenization(text)
        assert len(set(toks)) == 2 ** (len(text) - 1)
        for each in toks:
            assert ''.join(each) == text


if __name__ == '__main__':
    unittest.main()
