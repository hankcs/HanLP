import unittest

from hanlp_trie import Trie


class TestTrie(unittest.TestCase):
    def build_small_trie(self):
        return Trie({'商品': 'goods', '和': 'and', '和服': 'kimono', '服务': 'service', '务': 'business'})

    def assert_results_valid(self, text, results, trie):
        for begin, end, value in results:
            self.assertEqual(value, trie[text[begin:end]])

    def test_parse(self):
        trie = self.build_small_trie()
        text = '商品和服务'
        parse_result = trie.parse(text)
        self.assert_results_valid(text, parse_result, trie)
        self.assertEqual([(0, 2, 'goods'),
                          (2, 3, 'and'),
                          (2, 4, 'kimono'),
                          (3, 5, 'service'),
                          (4, 5, 'business')],
                         parse_result)

    def test_parse_longest(self):
        trie = self.build_small_trie()
        text = '商品和服务'
        parse_longest_result = trie.parse_longest(text)
        self.assert_results_valid(text, parse_longest_result, trie)
        self.assertEqual([(0, 2, 'goods'), (2, 4, 'kimono'), (4, 5, 'business')],
                         parse_longest_result)

    def test_items(self):
        trie = self.build_small_trie()
        items = list(trie.items())
        self.assertEqual([('商品', 'goods'), ('和', 'and'), ('和服', 'kimono'), ('服务', 'service'), ('务', 'business')], items)


if __name__ == '__main__':
    unittest.main()
