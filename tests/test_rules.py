# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-03-22 17:17
import unittest

from hanlp.utils.rules import split_sentence


class TestRules(unittest.TestCase):
    def test_eos(self):
        self.assertListEqual(list(split_sentence('叶')), ['叶'])
        self.assertListEqual(list(split_sentence('他说：“加油。”谢谢')), ['他说：“加油。”', '谢谢'])
        self.assertListEqual(list(split_sentence('Go to hankcs.com. Yes.')), ['Go to hankcs.com.', 'Yes.'])


if __name__ == '__main__':
    unittest.main()
