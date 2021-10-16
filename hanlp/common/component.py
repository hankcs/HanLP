# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-26 14:45
import inspect
from abc import ABC, abstractmethod
from typing import Any

from hanlp_common.configurable import Configurable


class Component(Configurable, ABC):
    @abstractmethod
    def predict(self, *args, **kwargs):
        """Predict on data. This is the base class for all components, including rule based and statistical ones.

        Args:
          *args: Any type of data subject to sub-classes
          **kwargs: Additional arguments

        Returns: Any predicted annotations.

        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def __call__(self, *args, **kwargs):
        """
        A shortcut for :func:`~hanlp.common.component.predict`.

        Args:
          *args: Any type of data subject to sub-classes
          **kwargs: Additional arguments

        Returns: Any predicted annotations.

        """
        return self.predict(*args, **kwargs)
