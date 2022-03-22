import hanlp
import unittest
from multiprocessing.dummy import Pool
from hanlp_common.document import Document

mtl = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH, devices=-1)


def tokenize(mtl, text):
    return mtl(text, tasks='tok/fine')['tok/fine']


class TestMultiTaskLearning(unittest.TestCase):
    def test_mtl_single_sent(self):
        doc: Document = mtl('商品和服务')
        self.assertSequenceEqual(doc['tok/fine'], ["商品", "和", "服务"])

    def test_mtl_multiple_sents(self):
        doc: Document = mtl(['商品和服务', '研究生命'])
        self.assertSequenceEqual(doc['tok/fine'], [
            ["商品", "和", "服务"],
            ["研究", "生命"]
        ])

    def test_mtl_empty_str(self):
        mtl('')
        mtl(' ')
        mtl([''])
        mtl([' '])
        mtl(['', ' '])
        mtl(['', ' ', 'good'])

    def test_skip_tok(self):
        pre_tokenized_sents = [
            ["商品和服务", '一个', '词'],
            ["研究", "生命"]
        ]
        doc: Document = mtl(pre_tokenized_sents, skip_tasks='tok*')
        self.assertSequenceEqual(doc['tok'], pre_tokenized_sents)

    def test_sdp_as_the_first_task(self):
        doc: Document = mtl(['人', '吃', '鱼'], tasks='sdp', skip_tasks='tok*')
        self.assertDictEqual(
            doc.to_dict(),
            {
                "sdp": [
                    [(2, "Agt")],
                    [(0, "Root")],
                    [(2, "Pat")]
                ],
                "tok": [
                    "人",
                    "吃",
                    "鱼"
                ]
            }
        )

    def test_threading(self):
        num_proc = 8
        with Pool(num_proc) as pool:
            results = pool.starmap(tokenize, [(mtl, '商品和服务')] * num_proc)
            self.assertSequenceEqual(results, [['商品', '和', '服务']] * num_proc)

    def test_emoji(self):
        self.assertSequenceEqual(mtl('( ͡° ͜ʖ ͡ °)你好', tasks='tok/fine')['tok/fine'],
                                 ["(", " ͡", "°", " ͜", "ʖ", " ͡ ", "°", ")", "你", "好"])
        mtl['tok/fine'].dict_combine = {'( ͡° ͜ʖ ͡ °)'}
        self.assertSequenceEqual(mtl('( ͡° ͜ʖ ͡ °)你好', tasks='tok/fine')['tok/fine'],
                                 ["( ͡° ͜ʖ ͡ °)", "你", "好"])

    def test_unicode_removed_by_hf(self):
        self.assertSequenceEqual(mtl('͡', tasks='tok/fine')['tok/fine'], ['͡'])

    def test_space(self):
        task = 'tok/fine'
        doc: Document = mtl('商品 和服务', tasks=task)
        self.assertSequenceEqual(doc[task], ["商品", "和", "服务"])
        mtl[task].dict_combine = {('iPad', 'Pro'), '2个空格'}
        self.assertSequenceEqual(mtl("如何评价iPad Pro ？iPad  Pro有2个空格", tasks=task)[task],
                                 ['如何', '评价', 'iPad Pro', '？', 'iPad  Pro', '有', '2个空格'])

    def test_transform(self):
        task = 'tok/fine'
        mtl[task].dict_force = {'用户ID'}
        self.assertSequenceEqual(mtl("我的用户ID跟你的用户id不同", tasks=task)[task],
                                 ['我', '的', '用户ID', '跟', '你', '的', '用户', 'id', '不同'])


if __name__ == '__main__':
    unittest.main()
