import unittest

from hanlp.common.structure import ConfigTracker


class MyClass(ConfigTracker):
    def __init__(self, i_need_this='yes') -> None:
        super().__init__(locals())


class TestConfigTracker(unittest.TestCase):
    def test_init(self):
        obj = MyClass()
        self.assertEqual(obj.config.get('i_need_this', None), 'yes')


if __name__ == '__main__':
    unittest.main()
