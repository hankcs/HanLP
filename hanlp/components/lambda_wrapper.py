# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-31 18:36
from typing import Callable, Any

from hanlp.common.component import Component
from hanlp.utils.reflection import class_path_of, object_from_class_path, str_to_type


class LambdaComponent(Component):
    def __init__(self, function: Callable) -> None:
        super().__init__()
        self.function = function
        self.meta['function'] = class_path_of(function)

    def predict(self, data: Any, **kwargs):
        unpack = kwargs.pop('_hanlp_unpack', None)
        if unpack:
            return self.function(*data, **kwargs)
        return self.function(data, **kwargs)

    @staticmethod
    def from_meta(meta: dict, **kwargs):
        cls = str_to_type(meta['class_path'])
        function = meta['function']
        function = object_from_class_path(function)
        return cls(function)
