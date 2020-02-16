# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 18:36
from typing import Callable, Any

from hanlp.common.component import Component
from hanlp_common.reflection import classpath_of, object_from_classpath, str_to_type


class LambdaComponent(Component):
    def __init__(self, function: Callable) -> None:
        super().__init__()
        self.config = {}
        self.function = function
        self.config['function'] = classpath_of(function)
        self.config['classpath'] = classpath_of(self)

    def predict(self, data: Any, **kwargs):
        unpack = kwargs.pop('_hanlp_unpack', None)
        if unpack:
            return self.function(*data, **kwargs)
        return self.function(data, **kwargs)

    @staticmethod
    def from_config(meta: dict, **kwargs):
        cls = str_to_type(meta['classpath'])
        function = meta['function']
        function = object_from_classpath(function)
        return cls(function)
