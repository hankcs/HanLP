# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-28 19:27
from typing import List

from phrasetree.tree import Tree

from hanlp_common.constant import EOS, BOS
from hanlp.common.dataset import TransformableDataset


class ConstituencyDataset(TransformableDataset):
    def load_file(self, filepath: str):
        with open(filepath) as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                yield {'constituency': Tree.fromstring(line)}


def unpack_tree_to_features(sample: dict):
    tree = sample.get('constituency', None)
    if tree:
        words, tags = zip(*tree.pos())
        chart = [[None] * (len(words) + 1) for _ in range(len(words) + 1)]
        for i, j, label in factorize(binarize(tree)[0]):
            # if no_subcategory:
            #     label = label.split('-')[0]
            chart[i][j] = label
        sample['token'] = [BOS] + list(words) + [EOS]
        sample['chart'] = chart
    return sample


def append_bos_eos(sample: dict):
    if '_con_token' not in sample:
        sample['_con_token'] = sample['token']
        sample['token'] = [BOS] + sample['token'] + [EOS]
    return sample


def remove_subcategory(sample: dict):
    tree: Tree = sample.get('constituency', None)
    if tree:
        for subtree in tree.subtrees():
            label = subtree.label()
            subtree.set_label(label.split('-')[0])
    return sample


def binarize(tree: Tree):
    r"""
    Conducts binarization over the tree.

    First, the tree is transformed to satisfy `Chomsky Normal Form (CNF)`_.
    Here we call :meth:`~tree.Tree.chomsky_normal_form` to conduct left-binarization.
    Second, all unary productions in the tree are collapsed.

    Args:
        tree (tree.Tree):
            The tree to be binarized.

    Returns:
        The binarized tree.

    Examples:
        >>> tree = Tree.fromstring('''
                                        (TOP
                                          (S
                                            (NP (_ She))
                                            (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                            (_ .)))
                                        ''')
        >>> print(Tree.binarize(tree))
        (TOP
          (S
            (S|<>
              (NP (_ She))
              (VP
                (VP|<> (_ enjoys))
                (S+VP (VP|<> (_ playing)) (NP (_ tennis)))))
            (S|<> (_ .))))

    .. _Chomsky Normal Form (CNF):
        https://en.wikipedia.org/wiki/Chomsky_normal_form
    """

    tree: Tree = tree.copy(True)
    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Tree):
            nodes.extend([child for child in node])
            if len(node) > 1:
                for i, child in enumerate(node):
                    if not isinstance(child[0], Tree):
                        node[i] = Tree(f"{node.label()}|<>", [child])
    tree.chomsky_normal_form('left', 0, 0)
    tree.collapse_unary()

    return tree


def factorize(tree, delete_labels=None, equal_labels=None):
    r"""
    Factorizes the tree into a sequence.
    The tree is traversed in pre-order.

    Args:
        tree (tree.Tree):
            The tree to be factorized.
        delete_labels (set[str]):
            A set of labels to be ignored. This is used for evaluation.
            If it is a pre-terminal label, delete the word along with the brackets.
            If it is a non-terminal label, just delete the brackets (don't delete childrens).
            In `EVALB`_, the default set is:
            {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
            Default: ``None``.
        equal_labels (dict[str, str]):
            The key-val pairs in the dict are considered equivalent (non-directional). This is used for evaluation.
            The default dict defined in `EVALB`_ is: {'ADVP': 'PRT'}
            Default: ``None``.

    Returns:
        The sequence of the factorized tree.

    Examples:
        >>> tree = Tree.fromstring('' (TOP
                                          (S
                                            (NP (_ She))
                                            (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                            (_ .)))
                                    '')
        >>> Tree.factorize(tree)
        [(0, 5, 'TOP'), (0, 5, 'S'), (0, 1, 'NP'), (1, 4, 'VP'), (2, 4, 'S'), (2, 4, 'VP'), (3, 4, 'NP')]
        >>> Tree.factorize(tree, delete_labels={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''})
        [(0, 5, 'S'), (0, 1, 'NP'), (1, 4, 'VP'), (2, 4, 'S'), (2, 4, 'VP'), (3, 4, 'NP')]

    .. _EVALB:
        https://nlp.cs.nyu.edu/evalb/
    """

    def track(tree, i):
        label = tree.label()
        if delete_labels is not None and label in delete_labels:
            label = None
        if equal_labels is not None:
            label = equal_labels.get(label, label)
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return (i + 1 if label is not None else i), []
        j, spans = i, []
        for child in tree:
            if isinstance(child, Tree):
                j, s = track(child, j)
                spans += s
        if label is not None and j > i:
            spans = [(i, j, label)] + spans
        return j, spans

    return track(tree, 0)[1]


def build_tree(tokens: List[str], sequence):
    r"""
    Builds a constituency tree from the sequence. The sequence is generated in pre-order.
    During building the tree, the sequence is de-binarized to the original format (i.e.,
    the suffixes ``|<>`` are ignored, the collapsed labels are recovered).

    Args:
        tokens :
            All tokens in a sentence.
        sequence (list[tuple]):
            A list of tuples used for generating a tree.
            Each tuple consits of the indices of left/right span boundaries and label of the span.

    Returns:
        A result constituency tree.

    Examples:
        >>> tree = Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], 'TOP')
        >>> sequence = [(0, 5, 'S'), (0, 4, 'S|<>'), (0, 1, 'NP'), (1, 4, 'VP'), (1, 2, 'VP|<>'),
                        (2, 4, 'S+VP'), (2, 3, 'VP|<>'), (3, 4, 'NP'), (4, 5, 'S|<>')]
        >>> print(Tree.build_tree(root, sequence))
        (TOP
          (S
            (NP (_ She))
            (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
            (_ .)))
    """
    if not tokens:  # User passed in [], which is the tokenized result of ''
        return Tree('TOP', [])
    tree = Tree('TOP', [Tree('_', [t]) for t in tokens])
    root = tree.label()
    leaves = [subtree for subtree in tree.subtrees() if not isinstance(subtree[0], Tree)]

    def track(node):
        i, j, label = next(node)
        if j == i + 1:
            children = [leaves[i]]
        else:
            children = track(node) + track(node)
        if label.endswith('|<>'):
            return children
        labels = label.split('+')
        tree = Tree(labels[-1], children)
        for label in reversed(labels[:-1]):
            tree = Tree(label, [tree])
        return [tree]

    return Tree(root, track(iter(sequence)))
