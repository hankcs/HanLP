import unittest

from hanlp_restful import HanLPClient


class TestClient(unittest.TestCase):

    def setUp(self) -> None:
        self.HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth=None)  # Fill in your auth

    def test_raw_text(self):
        text = '2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。'
        doc = self.HanLP.parse(text)

    def test_sents(self):
        text = ['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。',
                '阿婆主来到北京立方庭参观自然语义科技公司。']
        doc = self.HanLP(text)

    def test_tokens(self):
        tokens = [
            ["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多语种", "NLP", "技术", "。"],
            ["英", "首相", "与", "特朗普", "通", "电话", "讨论", "华为", "与", "苹果", "公司", "。"]
        ]
        doc = self.HanLP(tokens=tokens, tasks=['ner*', 'srl', 'dep'])

    def test_sents_mul(self):
        text = ['In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environment.',
                '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
                '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。']
        doc = self.HanLP.parse(text, language='mul')

    def test_tokenize(self):
        print(self.HanLP.tokenize('商品和服务。阿婆主来到北京立方庭参观自然语义科技公司'))
        print(self.HanLP.tokenize('商品和服务。阿婆主来到北京立方庭参观自然语义科技公司', coarse=True))
        print(self.HanLP.tokenize(['商品和服务。', '当下雨天地面积水分外严重']))
        print(self.HanLP.tokenize(
            ['In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environment.',
             '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
             '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。'], language='mul'))

    def test_coreference_resolution(self):
        print(self.HanLP.coreference_resolution('我姐送我她的猫。我很喜欢它。'))

    def test_text_style_transfer(self):
        print(self.HanLP.text_style_transfer('国家对中石油抱有很大的期望.', target_style='gov_doc'))
        print(self.HanLP.text_style_transfer('打工人，打工魂，打工都是人上人', target_style='gov_doc'))
        print(self.HanLP.text_style_transfer('我看到了窗户外面有白色的云和绿色的森林', target_style='modern_poetry'))


if __name__ == '__main__':
    unittest.main()
