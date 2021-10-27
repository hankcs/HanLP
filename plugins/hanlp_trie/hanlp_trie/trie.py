# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-04 23:46
from typing import Dict, Any, List, Tuple, Sequence, Union, Iterable, Optional


class Node(object):
    def __init__(self, value=None) -> None:
        """A node in a trie tree.

        Args:
            value: The value associated with this node.
        """
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
        """Transit the state of a Deterministic Finite Automata (DFA) with key.

        Args:
            key: A sequence of criterion (tokens or characters) used to transit to a new state.

        Returns:
            A new state if the transition succeeded, otherwise ``None``.

        """
        state = self
        for char in key:
            state = state._children.get(char)
            if state is None:
                break
        return state

    def _walk(self, prefix: Union[str, tuple], ordered=False):
        for char, child in sorted(self._children.items()) if ordered else self._children.items():
            prefix_new = prefix + (char if isinstance(prefix, str) else (char,))
            if child._value:
                yield prefix_new, child._value
            yield from child._walk(prefix_new)


class Trie(Node):
    def __init__(self, tokens: Optional[Union[Dict[str, Any], Iterable[str]]] = None) -> None:
        """A referential implementation of the trie (:cite:`10.1145/1457838.1457895`) structure. It stores a dict by
        assigning each key/value pair a :class:`~hanlp_trie.trie.Node` in a trie tree. It provides get/set/del/items
        methods just like a :class:`dict` does. Additionally, it also provides longest-prefix-matching and keywords
        lookup against a piece of text, which are very helpful in rule-based Natural Language Processing.

        Args:
            tokens: A set of keys or a dict mapping.
        """
        super().__init__()
        self._size = 0
        if tokens:
            if isinstance(tokens, dict):
                for k, v in tokens.items():
                    self[k] = v
            else:
                for k in tokens:
                    self[k] = True

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
        self._size += 1

    def __delitem__(self, key):
        state = self.transit(key)
        if state is not None:
            state._value = None
            self._size -= 1

    def update(self, dic: Dict[str, Any]):
        for k, v in dic.items():
            self[k] = v
        return self

    def parse(self, text: Sequence[str]) -> List[Tuple[int, int, Any]]:
        """Keywords lookup which takes a piece of text as input, and lookup all occurrences of keywords in it. These
        occurrences can overlap with each other.

        Args:
            text: A piece of text. In HanLP's design, it doesn't really matter whether this is a str or a list of str.
                The trie will transit on either types properly, which means a list of str simply defines a list of
                transition criteria while a str defines each criterion as a character.

        Returns:
            A tuple of ``(begin, end, value)``.
        """
        found = []
        for i in range(len(text)):
            state = self
            for j in range(i, len(text)):
                state = state.transit(text[j])
                if state:
                    if state._value is not None:
                        found.append((i, j + 1, state._value))
                else:
                    break
        return found

    def parse_longest(self, text: Sequence[str]) -> List[Tuple[int, int, Any]]:
        """Longest-prefix-matching which tries to match the longest keyword sequentially from the head of the text till
        its tail. By definition, the matches won't overlap with each other.

        Args:
            text: A piece of text. In HanLP's design, it doesn't really matter whether this is a str or a list of str.
                The trie will transit on either types properly, which means a list of str simply defines a list of
                transition criteria while a str defines each criterion as a character.

        Returns:
            A tuple of ``(begin, end, value)``.

        """
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
                    found.append((i, end, value))
                    i = end - 1
            i += 1
        return found

    def items(self, ordered=False, prefix=''):
        yield from self._walk(prefix, ordered)

    def __len__(self):
        return self._size

    def __bool__(self):
        return bool(len(self))
