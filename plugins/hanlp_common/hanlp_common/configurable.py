# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-16 22:24
from hanlp_common.reflection import str_to_type, classpath_of


class Configurable(object):
    @staticmethod
    def from_config(config: dict, **kwargs):
        """Build an object from config.

        Args:
          config: A ``dict`` holding parameters for its constructor. It has to contain a `classpath` key,
                    which has a classpath str as its value. ``classpath`` will determine the type of object
                    being deserialized.
          kwargs: Arguments not used.

        Returns: A deserialized object.

        """
        cls = config.get('classpath', None)
        assert cls, f'{config} doesn\'t contain classpath field'
        cls = str_to_type(cls)
        deserialized_config = dict(config)
        for k, v in config.items():
            if isinstance(v, dict) and 'classpath' in v:
                deserialized_config[k] = Configurable.from_config(v)
        if cls.from_config == Configurable.from_config:
            deserialized_config.pop('classpath')
            return cls(**deserialized_config)
        else:
            return cls.from_config(deserialized_config)


class AutoConfigurable(Configurable):
    @property
    def config(self) -> dict:
        """
        The config of this object, which are public properties. If any properties needs to be excluded from this config,
        simply declare it with prefix ``_``.
        """
        return dict([('classpath', classpath_of(self))] +
                    [(k, v.config if hasattr(v, 'config') else v)
                     for k, v in self.__dict__.items() if
                     not k.startswith('_')])

    def __repr__(self) -> str:
        return repr(self.config)
