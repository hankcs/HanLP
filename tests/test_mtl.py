import hanlp
import unittest
from multiprocessing.dummy import Pool
from hanlp_common.document import Document


def tokenize(mtl, text):
    return mtl(text, tasks='tok/fine')['tok/fine']


class TestMultiTaskLearning(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.mtl = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH, devices=-1)

    def test_mtl_single_sent(self):
        doc: Document = self.mtl('商品和服务')
        self.assertSequenceEqual(doc['tok/fine'], ["商品", "和", "服务"])

    def test_mtl_multiple_sents(self):
        doc: Document = self.mtl(['商品和服务', '研究生命'])
        self.assertSequenceEqual(doc['tok/fine'], [
            ["商品", "和", "服务"],
            ["研究", "生命"]
        ])

    def test_skip_tok(self):
        pre_tokenized_sents = [
            ["商品和服务", '一个', '词'],
            ["研究", "生命"]
        ]
        doc: Document = self.mtl(pre_tokenized_sents, skip_tasks='tok*')
        self.assertSequenceEqual(doc['tok'], pre_tokenized_sents)

    def test_threading(self):
        num_proc = 8
        with Pool(num_proc) as pool:
            results = pool.starmap(tokenize, [(self.mtl, '商品和服务')] * num_proc)
            self.assertSequenceEqual(results, [['商品', '和', '服务']] * num_proc)

    def test_emoji(self):
        self.assertSequenceEqual(self.mtl('( ͡° ͜ʖ ͡ °)你好', tasks='tok/fine')['tok/fine'],
                                 ["(", " ͡", "°", " ͜", "ʖ", " ͡ ", "°", ")", "你", "好"])
        self.mtl['tok/fine'].dict_combine = {'( ͡° ͜ʖ ͡ °)'}
        self.assertSequenceEqual(self.mtl('( ͡° ͜ʖ ͡ °)你好', tasks='tok/fine')['tok/fine'],
                                 ["( ͡° ͜ʖ ͡ °)", "你", "好"])


if __name__ == '__main__':
    unittest.main()
