# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-29 17:48
import json
from typing import Union, List, Optional, Dict, Any, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from hanlp_common.document import Document

try:
    # noinspection PyUnresolvedReferences
    import requests


    def _post(url, form: Dict[str, Any], headers: Dict[str, Any], timeout=10, verify=True) -> str:
        response = requests.post(url, json=form, headers=headers, timeout=timeout, verify=verify)
        if response.status_code != 200:
            raise HTTPError(url, response.status_code, response.text, response.headers, None)
        return response.text
except ImportError:
    import ssl


    def _post(url, form: Dict[str, Any], headers: Dict[str, Any], timeout=10, verify=True) -> str:
        request = Request(url, json.dumps(form).encode())
        for k, v in headers.items():
            request.add_header(k, v)
        ctx = None
        if not verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return urlopen(request, timeout=timeout, context=ctx).read().decode()


class HanLPClient(object):

    def __init__(self, url: str, auth: str = None, language=None, timeout=10, verify=True) -> None:
        """

        Args:
            url (str): An API endpoint to a service provider.
            auth (str): An auth key licenced from a service provider.
            language (str): The default language for each :func:`~hanlp_restful.HanLPClient.parse` call.
                Contact the service provider for the list of languages supported.
                Conventionally, ``zh`` is used for Chinese and ``mul`` for multilingual.
                Leave ``None`` to use the default language on server.
            timeout (int): Maximum waiting time in seconds for a request.
            verify (bool): ``True`` to enable SSL cert verification. You can also pass ``verify`` the path to a CA_BUNDLE
                file or directory with certificates of trusted CAs (``requests`` required).
        """
        super().__init__()
        self._language = language
        self._timeout = timeout
        self._url = url
        if auth is None:
            import os
            auth = os.getenv('HANLP_AUTH', None)
        self._auth = auth
        self._verify = verify

    def parse(self,
              text: Union[str, List[str]] = None,
              tokens: List[List[str]] = None,
              tasks: Optional[Union[str, List[str]]] = None,
              skip_tasks: Optional[Union[str, List[str]]] = None,
              language: str = None,
              ) -> Document:
        """
        Parse a piece of text.

        Args:
            text: A document (str), or a list of sentences (List[str]).
            tokens: A list of sentences where each sentence is a list of tokens.
            tasks: The tasks to predict. Use ``tasks=[...]`` to run selected tasks only. Dependent tasks will be
                automatically selected.
            skip_tasks: The tasks to skip. Use ``skip_tasks='tok/fine'`` to enable coarse tokenization for all tasks.
                Use ``tasks=['tok/coarse', ...]`` and ``skip_tasks='tok/fine'`` to enable coarse tokenization for
                selected tasks.
            language: The language of input text or tokens. ``None`` to use the default language on server.

        Returns:
            A :class:`~hanlp_common.document.Document`.

        Examples::

            # Use tasks=[...] to run selected tasks only
            HanLP('晓美焰来到自然语义科技公司', tasks=['pos', 'ner'])

            # Use skip_tasks='tok/fine' to enable coarse tokenization for all tasks
            HanLP('晓美焰来到自然语义科技公司', skip_tasks='tok/fine')

            # Use tasks=['tok/coarse', ...] and skip_tasks='tok/fine' to enable
            # coarse tokenization for selected tasks
            HanLP('晓美焰来到自然语义科技公司', tasks=['tok/coarse','pos'],skip_tasks='tok/fine')


        Raises:
            HTTPError: Any errors happening on the Internet side or the server side. Refer to the ``code`` and ``msg``
                of the exception for more details. A list of common errors :

        - ``400 Bad Request`` indicates that the server cannot process the request due to a client
          fault (e.g., text too long, language unsupported).
        - ``401 Unauthorized`` indicates that the request lacks **valid** ``auth`` credentials for the API.
        - ``422 Unprocessable Entity`` indicates that the content type of the request entity is not in
          proper json format.
        - ``429 Too Many Requests`` indicates the user has sent too many requests in a given
          amount of time ("rate limiting").

        """
        assert text or tokens, 'At least one of text or tokens has to be specified.'
        response = self._send_post_json(self._url + '/parse', {
            'text': text,
            'tokens': tokens,
            'tasks': tasks,
            'skip_tasks': skip_tasks,
            'language': language or self._language
        })
        return Document(response)

    def __call__(self,
                 text: Union[str, List[str]] = None,
                 tokens: List[List[str]] = None,
                 tasks: Optional[Union[str, List[str]]] = None,
                 skip_tasks: Optional[Union[str, List[str]]] = None,
                 language: str = None,
                 ) -> Document:
        """
        A shortcut of :meth:`~hanlp_restful.HanLPClient.parse`.
        """
        return self.parse(text, tokens, tasks, skip_tasks, language)

    def about(self) -> Dict[str, Any]:
        """Get the information about server and your client.

        Returns:
            A dict containing your rate limit and server version etc.

        """
        info = self._send_get_json(self._url + '/about', {})
        return Document(info)

    def _send_post(self, url, form: Dict[str, Any]):
        request = Request(url, json.dumps(form).encode())
        self._add_headers(request)
        return self._fire_request(request)

    def _fire_request(self, request):
        return urlopen(request, timeout=self._timeout).read().decode()

    def _send_post_json(self, url, form: Dict[str, Any]):
        headers = dict()
        if self._auth:
            headers['Authorization'] = f'Basic {self._auth}'
        return json.loads(_post(url, form, headers, self._timeout, verify=self._verify))

    def _send_get(self, url, form: Dict[str, Any]):
        request = Request(url + '?' + urlencode(form))
        self._add_headers(request)
        return self._fire_request(request)

    def _add_headers(self, request):
        if self._auth:
            request.add_header('Authorization', f'Basic {self._auth}')

    def _send_get_json(self, url, form: Dict[str, Any]):
        return json.loads(self._send_get(url, form))

    def text_style_transfer(self, text: Union[str, List[str]], target_style: str, language: str = None) \
            -> Union[str, List[str]]:
        """ Text style transfer aims to change the style of the input text to the target style while preserving its
        content.

        Args:
            text: Source text.
            target_style: Target style.
            language: The language of input text. ``None`` to use the default language.

        Returns:
            Text or a list of text of the target style.

        Examples::

            HanLP.text_style_transfer(['国家对中石油抱有很大的期望.', '要用创新去推动高质量的发展。'],
                                      target_style='gov_doc')
            # Output:
            [
                '国家对中石油寄予厚望。',
                '要以创新驱动高质量发展。'
            ]

            HanLP.text_style_transfer('我看到了窗户外面有白色的云和绿色的森林',
                                      target_style='modern_poetry')
            # Output:
            '我看见窗外的白云绿林'
        """
        response = self._send_post_json(self._url + '/text_style_transfer',
                                        {'text': text, 'target_style': target_style,
                                         'language': language or self._language})
        return response

    def semantic_textual_similarity(self, text: Union[Tuple[str, str], List[Tuple[str, str]]], language: str = None) \
            -> Union[float, List[float]]:
        """ Semantic textual similarity deals with determining how similar two pieces of texts are.

        Args:
            text: A pair or pairs of text.
            language: The language of input text. ``None`` to use the default language.

        Returns:
            Similarities.

        Examples::

            HanLP.semantic_textual_similarity([
                ('看图猜一电影名', '看图猜电影'),
                ('无线路由器怎么无线上网', '无线上网卡和无线路由器怎么用'),
                ('北京到上海的动车票', '上海到北京的动车票'),
            ])
            # Output:
            [
                0.9764469, # Similarity of ('看图猜一电影名', '看图猜电影')
                0.0,       # Similarity of ('无线路由器怎么无线上网', '无线上网卡和无线路由器怎么用')
                0.0034587  # Similarity of ('北京到上海的动车票', '上海到北京的动车票')
            ]
        """
        response = self._send_post_json(self._url + '/semantic_textual_similarity',
                                        {'text': text, 'language': language or self._language})
        return response

    def coreference_resolution(self, text: Optional[str] = None, tokens: Optional[List[List[str]]] = None,
                               speakers: Optional[List[str]] = None, language: Optional[str] = None) -> Union[
        Dict[str, Union[List[str], List[List[Tuple[str, int, int]]]]], List[List[Tuple[str, int, int]]]]:
        r""" Coreference resolution is the task of clustering mentions in text that refer to the same underlying
        real world entities.

        Args:
            text: A piece of text, usually a document without tokenization.
            tokens: A list of sentences where each sentence is a list of tokens.
            speakers: A list of speakers where each speaker is a ``str`` representing the speaker's ID, e.g., ``Tom``.
            language: The language of input text. ``None`` to use the default language.

        Returns:
            When ``text`` is specified, return the clusters and tokens. Otherwise just the clusters, In this case, you need to ``sum(tokens, [])`` in order to match the span indices with tokens

        Examples::

            HanLP.coreference_resolution('我姐送我她的猫。我很喜欢它。')
            # Output:
            {'clusters': [
                          [['我', 0, 1], ['我', 3, 4], ['我', 8, 9]], # 指代说话人
                          [['我姐', 0, 2], ['她', 4, 5]],             # 指代说话人的姐姐
                          [['她的猫', 4, 7], ['它', 11, 12]]],        # 指代说话人的姐姐的猫
             'tokens': ['我', '姐', '送', '我', '她', '的', '猫', '。',
                        '我', '很', '喜欢', '它', '。']}

            HanLP.coreference_resolution(
            tokens=[['我', '姐', '送', '我', '她', '的', '猫', '。'],
                    ['我', '很', '喜欢', '它', '。']])
            # Output:
                         [
                          [['我', 0, 1], ['我', 3, 4], ['我', 8, 9]], # 指代说话人
                          [['我姐', 0, 2], ['她', 4, 5]],             # 指代说话人的姐姐
                          [['她的猫', 4, 7], ['它', 11, 12]]],        # 指代说话人的姐姐的猫

        .. image:: https://file.hankcs.com/img/coref_demo_small.png
            :alt: Coreference resolution visualization
        """
        response = self._send_post_json(self._url + '/coreference_resolution',
                                        {'text': text, 'tokens': tokens, 'speakers': speakers,
                                         'language': language or self._language})
        return response

    def tokenize(self, text: Union[str, List[str]], coarse: Optional[bool] = None, language=None) -> List[List[str]]:
        """ Split a document into sentences and tokenize them. Note that it is always faster to tokenize a whole
        document than to tokenize each sentence one by one. So avoid calling this method sentence by sentence but put
        sentences into a ``list`` and pass them to the ``text`` argument.

        Args:
            text: A document (``str``), or a list of sentences (``List[str]``).
            coarse: Whether to perform coarse-grained or fine-grained tokenization.
            language: The language of input text. ``None`` to use the default language.

        Returns:
            A list of tokenized sentences.

        Examples::

            # Avoid tokenizing sentence by sentence, it is expensive:
            HanLP.tokenize('商品和服务。')
            [['商品', '和', '服务', '。']]
            HanLP.tokenize('阿婆主来到北京立方庭参观自然语义科技公司')
            [['阿婆主', '来到', '北京', '立方庭', '参观', '自然', '语义', '科技', '公司']]

            # Instead, the following codes are much faster:
            HanLP.tokenize('商品和服务。阿婆主来到北京立方庭参观自然语义科技公司')
            [['商品', '和', '服务', '。'],
             ['阿婆主', '来到', '北京', '立方庭', '参观', '自然', '语义', '科技', '公司']]

            # To tokenize with coarse-grained standard:
            HanLP.tokenize('商品和服务。阿婆主来到北京立方庭参观自然语义科技公司', coarse=True)
            [['商品', '和', '服务', '。'],
             ['阿婆主', '来到', '北京', '立方庭', '参观', '自然语义科技公司']]

            # To tokenize pre-segmented sentences:
            HanLP.tokenize(['商品和服务。', '当下雨天地面积水分外严重'])
            [['商品', '和', '服务', '。'],
             ['当', '下雨天', '地面', '积水', '分', '外', '严重']]

            # Multilingual tokenization by specifying language='mul':
            HanLP.tokenize(
                ['In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques
                 'to production environment.',
                 '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
                 '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。'], language='mul')
            [['In', '2021', ',', 'HanLPv2.1', 'delivers', 'state-of-the-art', 'multilingual',
              'NLP', 'techniques', 'to', 'production', 'environment', '.'],
             ['2021', '年', '、', 'HanLPv2.1', 'は', '次', '世代', 'の', '最', '先端', '多',
              '言語', 'NLP', '技術', 'を', '本番', '環境', 'に', '導入', 'します', '。'],
             ['2021', '年', 'HanLPv2.1', '为', '生产', '环境', '带来', '次世代', '最', '先进的',
              '多', '语种', 'NLP', '技术', '。']]
        """
        language = language or self._language
        if coarse and language and language != 'zh':
            raise NotImplementedError(f'Coarse tokenization not supported for {language}. Please set language="zh".')
        doc = self.parse(text=text, tasks='tok/coarse' if coarse is True else 'tok', language=language)
        return next(iter(doc.values()))

    def abstract_meaning_representation(self,
                                        text: Union[str, List[str]] = None,
                                        tokens: List[List[str]] = None,
                                        language: str = None,
                                        visualization: str = None,
                                        ) -> List[Dict]:
        """Abstract Meaning Representation (AMR) captures “who is doing what to whom” in a sentence. Each sentence is
        represented as a rooted, directed, acyclic graph consisting of nodes (concepts) and edges (relations).

        Args:
            text: A document (str), or a list of sentences (List[str]).
            tokens: A list of sentences where each sentence is a list of tokens.
            language: The language of input text or tokens. ``None`` to use the default language on server.
            visualization: Set to `dot` or `svg` to obtain coresspodning visualization.

        Returns:
            Graphs in meaning represenation format.

        Examples::

            HanLP.abstract_meaning_representation('男孩希望女孩相信他。')
            HanLP.abstract_meaning_representation('The boy wants the girl to believe him.',
                                                  language='en')

        .. image:: https://hanlp.hankcs.com/backend/v2/amr_svg?tokens=%E7%94%B7%E5%AD%A9%20%E5%B8%8C%E6%9C%9B%20%E5%A5%B3%E5%AD%A9%20%E7%9B%B8%E4%BF%A1%20%E4%BB%96%20%E3%80%82&language=zh&scale=1
            :alt: Abstract Meaning Representation

        .. image:: https://hanlp.hankcs.com/backend/v2/amr_svg?tokens=The%20boy%20wants%20the%20girl%20to%20believe%20him%20.&language=en&scale=1
            :alt: Abstract Meaning Representation

        """
        assert text or tokens, 'At least one of text or tokens has to be specified.'
        return self._send_post_json(self._url + '/abstract_meaning_representation', {
            'text': text,
            'tokens': tokens,
            'language': language or self._language,
            'visualization': visualization,
        })

    def keyphrase_extraction(
            self,
            text: str,
            topk: int = 10,
            language: str = None,
    ) -> Dict[str, float]:
        """ Keyphrase extraction aims to identify keywords or phrases reflecting the main topics of a document.

        Args:
            text: The text content of the document. Preferably the concatenation of the title and the content.
            topk: The number of top-K ranked keywords or keyphrases.
            language: The language of input text or tokens. ``None`` to use the default language on server.

        Returns:
            A dictionary containing each keyword or keyphrase and its ranking score :math:`s`, :math:`s \in [0, 1]`.

        Examples::

            HanLP.keyphrase_extraction(
                '自然语言处理是一门博大精深的学科，掌握理论才能发挥出HanLP的全部性能。 '
                '《自然语言处理入门》是一本配套HanLP的NLP入门书，助你零起点上手自然语言处理。', topk=3)
            # Output:
            {'自然语言处理': 0.800000011920929,
             'HanLP的全部性能': 0.5258446335792542,
             '一门博大精深的学科': 0.421421080827713}
        """
        assert text, 'Text has to be specified.'
        return self._send_post_json(self._url + '/keyphrase_extraction', {
            'text': text,
            'language': language or self._language,
            'topk': topk,
        })

    def extractive_summarization(
            self,
            text: str,
            topk: int = 3,
            language: str = None,
    ) -> Dict[str, float]:
        """ Single document summarization is the task of selecting a subset of the sentences which best
        represents a summary of the document, with a balance of salience and redundancy.

        Args:
            text: The text content of the document.
            topk: The maximum number of top-K ranked sentences. Note that due to Trigram Blocking tricks, the actual
                number of returned sentences could be less than ``topk``.
            language: The language of input text or tokens. ``None`` to use the default language on server.

        Returns:
            A dictionary containing each sentence and its ranking score :math:`s \in [0, 1]`.

        Examples::

            HanLP.extractive_summarization('''
            据DigiTimes报道，在上海疫情趋缓，防疫管控开始放松后，苹果供应商广达正在逐步恢复其中国工厂的MacBook产品生产。
            据供应链消息人士称，生产厂的订单拉动情况正在慢慢转强，这会提高MacBook Pro机型的供应量，并缩短苹果客户在过去几周所经历的延长交货时间。
            仍有许多苹果笔记本用户在等待3月和4月订购的MacBook Pro机型到货，由于苹果的供应问题，他们的发货时间被大大推迟了。
            据分析师郭明錤表示，广达是高端MacBook Pro的唯一供应商，自防疫封控依赖，MacBook Pro大部分型号交货时间增加了三到五周，
            一些高端定制型号的MacBook Pro配置要到6月底到7月初才能交货。
            尽管MacBook Pro的生产逐渐恢复，但供应问题预计依然影响2022年第三季度的产品销售。
            苹果上周表示，防疫措施和元部件短缺将继续使其难以生产足够的产品来满足消费者的强劲需求，这最终将影响苹果6月份的收入。
            ''')
            # Output:
            {'据DigiTimes报道，在上海疫情趋缓，防疫管控开始放松后，苹果供应商广达正在逐步恢复其中国工厂的MacBook产品生产。': 0.9999,
             '仍有许多苹果笔记本用户在等待3月和4月订购的MacBook Pro机型到货，由于苹果的供应问题，他们的发货时间被大大推迟了。': 0.5800,
             '尽管MacBook Pro的生产逐渐恢复，但供应问题预计依然影响2022年第三季度的产品销售。': 0.5422}
        """
        assert text, 'Text has to be non-empty.'
        return self._send_post_json(self._url + '/extractive_summarization', {
            'text': text,
            'language': language or self._language,
            'topk': topk,
        })

    def grammatical_error_correction(self, text: Union[str, List[str]], language: str = None) \
            -> Union[str, List[str]]:
        """ Grammatical Error Correction (GEC) is the task of correcting different kinds of errors in text such as
        spelling, punctuation, grammatical, and word choice errors.

        Args:
            text: Text potentially containing different kinds of errors such as spelling, punctuation,
                grammatical, and word choice errors.
            language: The language of input text. ``None`` to use the default language.

        Returns:
            Corrected text.

        Examples::

            HanLP.grammatical_error_correction(['每个青年都应当有远大的报复。',
                                                '有的同学对语言很兴趣。'])
            # Output:
            [
                '每个青年都应当有远大的抱负。',
                '有的同学对语言很有兴趣。'
            ]

        """
        response = self._send_post_json(self._url + '/grammatical_error_correction',
                                        {'text': text,
                                         'language': language or self._language})
        return response

    def text_classification(self, text: Union[str, List[str]], model, topk=False, prob=False) -> Union[
        str, Dict[str, float], List[Union[str, Dict[str, float]]]]:
        """
        Text classification is the task of assigning a sentence or document an appropriate category.
        The categories depend on the chosen dataset and can range from topics.

        Args:
            text: A document or a list of documents.
            model: The model to use for prediction.
            topk: ``True`` or ``int`` to return the top-k labels.
            prob: Return also probabilities.

        Returns:

            Classification results.
        """
        response = self._send_post_json(self._url + '/text_classification',
                                        {'text': text, 'model': model, 'topk': topk, 'prob': prob})
        return response

    def language_identification(self, text: Union[str, List[str]], topk=False, prob=False) -> Union[
        str, Dict[str, float], List[Union[str, Dict[str, float]]]]:
        """
        Identify the language of a given text.

        Args:
            text: A document or a list of documents.
            topk: ``True`` or ``int`` to return the top-k languages.
            prob: Return also probabilities.

        Returns:

            Identified language in `ISO 639-1 codes`_.

        Examples::

            HanLP.language_identification(
            'In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques.')
            'en'
            lang, prob = HanLP.language_identification(
            '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。', prob=True)
            ('ja', 0.9976244568824768)
            HanLP.language_identification(
            '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', topk=2)
            ['zh', 'ja']
            HanLP.language_identification(
            '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', topk=3, prob=True)
            {'zh': 0.3952908217906952, 'en': 0.37189167737960815, 'ja': 0.056213412433862686}

        .. _ISO 639-1 codes:
           https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
        """
        return self.text_classification(text, 'lid', topk, prob)
