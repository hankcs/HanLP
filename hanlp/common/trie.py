# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-04 23:46
from typing import Dict, Any, List, Tuple, Union


class Node(object):
    def __init__(self, value=None) -> None:
        self._children = {}
        self._value = value

    def _add_child(self, char, value, overwrite=False):
        child = self._children.get(char)
        if child is None:
            child = Node(value)
            self._children[char] = child
        elif overwrite:
            child._value = value
        return child

    def transit(self, key):
        state = self
        for char in key:
            state = state._children.get(char)
            if state is None:
                break
        return state


class Trie(Node):

    def __contains__(self, key):
        return self[key] is not None

    def __getitem__(self, key):
        state = self.transit(key)
        if state is None:
            return None
        return state._value

    def __setitem__(self, key, value):
        state = self
        for i, char in enumerate(key):
            if i < len(key) - 1:
                state = state._add_child(char, None, False)
            else:
                state = state._add_child(char, value, True)

    def __delitem__(self, key):
        state = self.transit(key)
        if state is not None:
            state._value = None

    def update(self, dic: Dict[str, Any]):
        for k, v in dic.items():
            self[k] = v
        return self

    def parse_longest(self, text: str) -> List[Tuple[str, Any, int, int]]:
        found = []
        for i in range(len(text)):
            state = self.transit(text[i])
            if state:
                to = i + 1
                end = to
                value = state._value
                for to in range(i + 1, len(text)):
                    state = state.transit(text[to])
                    if not state:
                        break
                    if state._value is not None:
                        value = state._value
                        end = to + 1
                if value is not None:
                    found.append((text[i:end], value, i, end))
        return found
