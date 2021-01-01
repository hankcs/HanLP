# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-16 22:38
import json
import os
import pickle
import sys


def save_pickle(item, path):
    with open(path, 'wb') as f:
        pickle.dump(item, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(item: dict, path: str, ensure_ascii=False, cls=None, default=lambda o: repr(o), indent=2):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(item, out, ensure_ascii=ensure_ascii, indent=indent, cls=cls, default=default)


def load_json(path):
    with open(path, encoding='utf-8') as src:
        return json.load(src)


def filename_is_json(filename):
    filename, file_extension = os.path.splitext(filename)
    return file_extension in ['.json', '.jsonl']


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)