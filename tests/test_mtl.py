import unittest
import hanlp
from hanlp_common.document import Document


class TestMultiTaskLearning(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.mtl = hanlp.load(hanlp.pretrained.mtl.OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH, devices=-1)

    def test_mtl_single_sent(self):
        doc: Document = self.mtl('商品和服务')
        self.assertSequenceEqual(doc['tok'], ["商品", "和", "服务"])

    def test_mtl_multiple_sents(self):
        doc: Document = self.mtl(['商品和服务', '研究生命'])
        self.assertSequenceEqual(doc['tok'], [
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


if __name__ == '__main__':
    unittest.main()
