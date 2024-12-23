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
        print(self.HanLP.tokenize('奈須きのこは1973年11月28日に千葉県円空山で生まれ、ゲーム制作会社「ノーツ」の設立者だ。',
                                  language='ja', coarse=True))
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

    def test_abstract_meaning_representation(self):
        print(self.HanLP.abstract_meaning_representation('男孩希望女孩相信他。'))
        print(self.HanLP.abstract_meaning_representation('男孩希望女孩相信他。', visualization='dot'))
        print(self.HanLP.abstract_meaning_representation('男孩希望女孩相信他。', visualization='svg'))
        print(self.HanLP.abstract_meaning_representation(tokens=[['男孩', '希望', '女孩', '相信', '他', '。']]))
        print(self.HanLP.abstract_meaning_representation('The boy wants the girl to believe him.', language='en'))

    def test_keyphrase_extraction(self):
        print(self.HanLP.keyphrase_extraction(
            '自然语言处理是一门博大精深的学科，掌握理论才能发挥出HanLP的全部性能。 '
            '《自然语言处理入门》是一本配套HanLP的NLP入门书，助你零起点上手自然语言处理。', topk=3))

    def test_extractive_summarization(self):
        text = '''
        据DigiTimes报道，在上海疫情趋缓，防疫管控开始放松后，苹果供应商广达正在逐步恢复其中国工厂的MacBook产品生产。
        据供应链消息人士称，生产厂的订单拉动情况正在慢慢转强，这会提高MacBook Pro机型的供应量，并缩短苹果客户在过去几周所经历的延长交货时间。
        仍有许多苹果笔记本用户在等待3月和4月订购的MacBook Pro机型到货，由于苹果的供应问题，他们的发货时间被大大推迟了。
        据分析师郭明錤表示，广达是高端MacBook Pro的唯一供应商，自防疫封控依赖，MacBook Pro大部分型号交货时间增加了三到五周，
        一些高端定制型号的MacBook Pro配置要到6月底到7月初才能交货。
        尽管MacBook Pro的生产逐渐恢复，但供应问题预计依然影响2022年第三季度的产品销售。
        苹果上周表示，防疫措施和元部件短缺将继续使其难以生产足够的产品来满足消费者的强劲需求，这最终将影响苹果6月份的收入。
            '''
        print(self.HanLP.extractive_summarization(text))


if __name__ == '__main__':
    unittest.main()
