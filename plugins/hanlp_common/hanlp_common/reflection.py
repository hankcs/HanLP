# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 16:41
import importlib
import inspect
import threading
from contextlib import contextmanager


_classpath_policy = threading.local()


class UntrustedClasspathError(ValueError):
    """A model attempted to resolve Python code outside HanLP's packages."""


@contextmanager
def trusted_classpaths(trust_remote_code=False):
    """Restrict reflective model loading, inheriting the outer load's policy.

    Outside model loading, reflection retains its existing general-purpose
    behavior. Nested loads cannot relax an existing restricted policy.
    """
    previous = getattr(_classpath_policy, 'trust_remote_code', None)
    _classpath_policy.trust_remote_code = bool(trust_remote_code) if previous is None else previous
    try:
        yield
    finally:
        _classpath_policy.trust_remote_code = previous


def _is_hanlp_module(module):
    return isinstance(module, str) and any(module == root or module.startswith(root + '.')
                                          for root in ('hanlp', 'hanlp_common', 'hanlp_trie'))


def classpath_of(obj) -> str:
    """get the full class path of object

    Args:
      obj: return:

    Returns:

    """
    if inspect.isfunction(obj):
        return module_path_of(obj)
    return "{0}.{1}".format(obj.__class__.__module__, obj.__class__.__name__)


def module_path_of(func) -> str:
    return inspect.getmodule(func).__name__ + '.' + func.__name__


def object_from_classpath(classpath, **kwargs):
    classpath = str_to_type(classpath)
    if inspect.isfunction(classpath):
        return classpath
    return classpath(**kwargs)


def str_to_type(classpath):
    """convert class path in str format to a type

    Args:
      classpath: class path

    Returns:
      type

    """
    module_name, class_name = classpath.rsplit(".", 1)
    restricted = getattr(_classpath_policy, 'trust_remote_code', None) is False
    if restricted and not _is_hanlp_module(module_name):
        raise UntrustedClasspathError(
            'Model classpath {!r} is outside HanLP. Only load trusted custom models with '
            'hanlp.load(..., trust_remote_code=True).'.format(classpath))
    cls = getattr(importlib.import_module(module_name), class_name)
    # A HanLP module may re-export a third-party class or function. Check its
    # defining module as well so an alias does not bypass the trust boundary.
    if restricted and not _is_hanlp_module(getattr(cls, '__module__', None)):
        raise UntrustedClasspathError('Model classpath {!r} resolves to code outside HanLP.'.format(classpath))
    return cls


def type_to_str(type_object) -> str:
    """convert a type object to class path in str format

    Args:
      type_object: type

    Returns:
      class path

    """
    cls_name = str(type_object)
    assert cls_name.startswith("<class '"), 'illegal input'
    cls_name = cls_name[len("<class '"):]
    assert cls_name.endswith("'>"), 'illegal input'
    cls_name = cls_name[:-len("'>")]
    return cls_name
