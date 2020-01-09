import unittest

import hanlp


class TestTransformerNamedEntityRecognizer(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)

    def test_unk_token(self):
        self.recognizer([list('孽债 （上海话）')])


if __name__ == '__main__':
    unittest.main()
