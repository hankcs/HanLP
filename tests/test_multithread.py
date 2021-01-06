import unittest
from multiprocessing.dummy import Pool

import hanlp

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH, devices=-1)


def tokenize(text):
    return HanLP(text, tasks='tok/fine')['tok/fine']


class TestMultiThreading(unittest.TestCase):
    def test_threading(self):
        num_proc = 8
        with Pool(num_proc) as pool:
            results = pool.map(tokenize, ['商品和服务'] * num_proc)
            self.assertSequenceEqual(results, [['商品', '和', '服务']] * num_proc)


if __name__ == '__main__':
    unittest.main()
