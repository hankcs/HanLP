# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 16:41
import importlib
import inspect


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
    cls = getattr(importlib.import_module(module_name), class_name)
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
