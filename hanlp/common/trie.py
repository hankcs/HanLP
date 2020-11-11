# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-04 23:46
from typing import Dict, Any, List, Tuple, Iterable, Sequence, Union, Set


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
    def __init__(self, tokens: Union[Dict[str, Any], Set[str]] = None) -> None:
        super().__init__()
        if tokens:
            if isinstance(tokens, set):
                for k in tokens:
                    self[k] = True
            else:
                for k, v in tokens.items():
                    self[k] = v

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

    def parse(self, text: Sequence[str]) -> List[Tuple[Union[str, Sequence[str]], Any, int, int]]:
        found = []
        for i in range(len(text)):
            state = self
            for j in range(i, len(text)):
                state = state.transit(text[j])
                if state:
                    if state._value is not None:
                        found.append((text[i: j + 1], state._value, i, j + 1))
                else:
                    break
        return found

    def parse_longest(self, text: Sequence[str]) -> List[Tuple[Union[str, Sequence[str]], Any, int, int]]:
        found = []
        i = 0
        while i < len(text):
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
                    i = end - 1
            i += 1
        return found
