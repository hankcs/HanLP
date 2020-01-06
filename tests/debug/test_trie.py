# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-05 22:47
from unittest import TestCase

from hanlp.common.trie import Trie


class TestTrie(TestCase):

    def test_transit(self):
        trie = self.create_trie()
        state = trie.transit('自然')
        self.assertEqual(2, len(state._children))
        self.assertTrue('自然' in trie)
        self.assertEqual('nature', trie['自然'])
        del trie['自然']
        self.assertFalse('自然' in trie)

    @staticmethod
    def create_trie():
        trie = Trie()
        trie['自然'] = 'nature'
        trie['自然人'] = 'human'
        trie['自然语言'] = 'language'
        trie['自语'] = 'talk	to oneself'
        trie['入门'] = 'introduction'
        return trie

    def test_parse_longest(self):
        trie = self.create_trie()
        trie.parse_longest('《自然语言处理入门》出版了')
