# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-26 14:58
from typing import Dict

from hanlp_common.configurable import Configurable
from hanlp_common.reflection import classpath_of
from hanlp_common.structure import SerializableDict


class ConfigTracker(Configurable):

    def __init__(self, locals_: Dict, exclude=('kwargs', 'self', '__class__', 'locals_')) -> None:
        """This base class helps sub-classes to capture their arguments passed to ``__init__``, and also their types so
        that they can be deserialized from a config in dict form.

        Args:
            locals_: Obtained by :meth:`locals`.
            exclude: Arguments to be excluded.

        Examples:
            >>> class MyClass(ConfigTracker):
            >>>     def __init__(self, i_need_this='yes') -> None:
            >>>         super().__init__(locals())
            >>> obj = MyClass()
            >>> print(obj.config)
            {'i_need_this': 'yes', 'classpath': 'test_config_tracker.MyClass'}

        """
        if 'kwargs' in locals_:
            locals_.update(locals_['kwargs'])
        self.config = SerializableDict(
            (k, v.config if hasattr(v, 'config') else v) for k, v in locals_.items() if k not in exclude)
        self.config['classpath'] = classpath_of(self)


class History(object):
    def __init__(self):
        """ A history of training context. It records how many steps have passed and provides methods to decide whether
        an update should be performed, and to caculate number of training steps given dataloader size and
        ``gradient_accumulation``.
        """
        self.num_mini_batches = 0

    def step(self, gradient_accumulation):
        """ Whether the training procedure should perform an update.

        Args:
            gradient_accumulation: Number of batches per update.

        Returns:
            bool: ``True`` to update.
        """
        self.num_mini_batches += 1
        return self.num_mini_batches % gradient_accumulation == 0

    def num_training_steps(self, num_batches, gradient_accumulation):
        """ Caculate number of training steps.

        Args:
            num_batches: Size of dataloader.
            gradient_accumulation: Number of batches per update.

        Returns:

        """
        return len(
            [i for i in range(self.num_mini_batches + 1, self.num_mini_batches + num_batches + 1) if
             i % gradient_accumulation == 0])
