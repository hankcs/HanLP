# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 16:41
import importlib
import inspect


def class_path_of(obj) -> str:
    """
    get the full class path of object
    :param obj:
    :return:
    """
    if inspect.isfunction(obj):
        return module_path_of(obj)
    return "{0}.{1}".format(obj.__class__.__module__, obj.__class__.__name__)


def module_path_of(func) -> str:
    return inspect.getmodule(func).__name__ + '.' + func.__name__


def object_from_class_path(class_path, **kwargs):
    class_path = str_to_type(class_path)
    if inspect.isfunction(class_path):
        return class_path
    return class_path(**kwargs)


def str_to_type(classpath):
    """
    convert class path in str format to a type
    :param classpath: class path
    :return: type
    """
    module_name, class_name = classpath.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls


def type_to_str(type_object) -> str:
    """
    convert a type object to class path in str format
    :param type_object: type
    :return: class path
    """
    cls_name = str(type_object)
    assert cls_name.startswith("<class '"), 'illegal input'
    cls_name = cls_name[len("<class '"):]
    assert cls_name.endswith("'>"), 'illegal input'
    cls_name = cls_name[:-len("'>")]
    return cls_name
