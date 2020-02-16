# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-19 20:56
import json
from collections import OrderedDict

from hanlp_common.io import filename_is_json, save_pickle, load_pickle, save_json, load_json


class Serializable(object):
    """A super class for save/load operations."""

    def save(self, path, fmt=None):
        if not fmt:
            if filename_is_json(path):
                self.save_json(path)
            else:
                self.save_pickle(path)
        elif fmt in ['json', 'jsonl']:
            self.save_json(path)
        else:
            self.save_pickle(path)

    def load(self, path, fmt=None):
        if not fmt:
            if filename_is_json(path):
                self.load_json(path)
            else:
                self.load_pickle(path)
        elif fmt in ['json', 'jsonl']:
            self.load_json(path)
        else:
            self.load_pickle(path)

    def save_pickle(self, path):
        """Save to path

        Args:
          path:

        Returns:


        """
        save_pickle(self, path)

    def load_pickle(self, path):
        """Load from path

        Args:
          path(str): file path

        Returns:


        """
        item = load_pickle(path)
        return self.copy_from(item)

    def save_json(self, path):
        save_json(self.to_dict(), path)

    def load_json(self, path):
        item = load_json(path)
        return self.copy_from(item)

    # @abstractmethod
    def copy_from(self, item):
        self.__dict__ = item.__dict__
        # raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def to_json(self, ensure_ascii=False, indent=2, sort=False) -> str:
        d = self.to_dict()
        if sort:
            d = OrderedDict(sorted(d.items()))
        return json.dumps(d, ensure_ascii=ensure_ascii, indent=indent, default=lambda o: repr(o))

    def to_dict(self) -> dict:
        return self.__dict__


class SerializableDict(Serializable, dict):

    def save_json(self, path):
        save_json(self, path)

    def copy_from(self, item):
        if isinstance(item, dict):
            self.clear()
            self.update(item)

    def __getattr__(self, key):
        if key.startswith('__'):
            return dict.__getattr__(key)
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def to_dict(self) -> dict:
        return self