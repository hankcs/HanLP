import unittest
import hanlp


class TestPipeLine(unittest.TestCase):
    def test_copy(self):
        pipe = hanlp.pipeline().append(hanlp.utils.rules.split_sentence)
        copied_pipe = pipe.copy()
        test_text = "今天天气真好。我要去散步。"
        assert pipe is not copied_pipe
        copied_pipe.append(lambda sent: "".join(sent))
        assert pipe(test_text) != copied_pipe(test_text)

if __name__ == '__main__':
    unittest.main()
