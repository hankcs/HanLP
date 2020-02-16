# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-04-02 23:20
from collections import defaultdict
from hanlp.components.parsers.chu_liu_edmonds import decode_mst
import numpy as np


class Tarjan:
    """Computes Tarjan's algorithm for finding strongly connected components (cycles) of a graph"""

    def __init__(self, prediction, tokens):
        """

        Parameters
        ----------
        prediction : numpy.ndarray
            a predicted dependency tree where prediction[dep_idx] = head_idx
        tokens : numpy.ndarray
            the tokens we care about (i.e. exclude _GO, _EOS, and _PAD)
        """
        self._edges = defaultdict(set)
        self._vertices = set((0,))
        for dep, head in enumerate(prediction[tokens]):
            self._vertices.add(dep + 1)
            self._edges[head].add(dep + 1)
        self._indices = {}
        self._lowlinks = {}
        self._onstack = defaultdict(lambda: False)
        self._SCCs = []

        index = 0
        stack = []
        for v in self.vertices:
            if v not in self.indices:
                self.strongconnect(v, index, stack)

    # =============================================================
    def strongconnect(self, v, index, stack):
        """

        Args:
          v: 
          index: 
          stack: 

        Returns:

        """

        self._indices[v] = index
        self._lowlinks[v] = index
        index += 1
        stack.append(v)
        self._onstack[v] = True
        for w in self.edges[v]:
            if w not in self.indices:
                self.strongconnect(w, index, stack)
                self._lowlinks[v] = min(self._lowlinks[v], self._lowlinks[w])
            elif self._onstack[w]:
                self._lowlinks[v] = min(self._lowlinks[v], self._indices[w])

        if self._lowlinks[v] == self._indices[v]:
            self._SCCs.append(set())
            while stack[-1] != v:
                w = stack.pop()
                self._onstack[w] = False
                self._SCCs[-1].add(w)
            w = stack.pop()
            self._onstack[w] = False
            self._SCCs[-1].add(w)
        return

    # ======================
    @property
    def edges(self):
        return self._edges

    @property
    def vertices(self):
        return self._vertices

    @property
    def indices(self):
        return self._indices

    @property
    def SCCs(self):
        return self._SCCs


class UnionFind(object):

    def __init__(self, n) -> None:
        super().__init__()
        self.parent = [x for x in range(n)]
        self.height = [0] * n

    def find(self, x):
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.height[x] < self.height[y]:
            self.parent[x] = y
        else:
            self.parent[y] = x
            if self.height[x] == self.height[y]:
                self.height[x] += 1

    def same(self, x, y):
        return self.find(x) == self.find(y)


def tarjan(parse_probs, length, tokens_to_keep, ensure_tree=True):
    """Adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/models/nn.py

    Args:
      parse_probs(NDArray): seq_len x seq_len, the probability of arcs
      length(NDArray): sentence length including ROOT
      tokens_to_keep(NDArray): mask matrix
      ensure_tree:  (Default value = True)

    Returns:

    
    """
    if ensure_tree:
        parse_preds, parse_probs, tokens = unique_root(parse_probs, tokens_to_keep, length)
        # remove cycles
        tarjan = Tarjan(parse_preds, tokens)
        for SCC in tarjan.SCCs:
            if len(SCC) > 1:
                dependents = set()
                to_visit = set(SCC)
                while len(to_visit) > 0:
                    node = to_visit.pop()
                    if not node in dependents:
                        dependents.add(node)
                        to_visit.update(tarjan.edges[node])
                # The indices of the nodes that participate in the cycle
                cycle = np.array(list(SCC))
                # The probabilities of the current heads
                old_heads = parse_preds[cycle]
                old_head_probs = parse_probs[cycle, old_heads]
                # Set the probability of depending on a non-head to zero
                non_heads = np.array(list(dependents))
                parse_probs[np.repeat(cycle, len(non_heads)), np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
                # Get new potential heads and their probabilities
                new_heads = np.argmax(parse_probs[cycle][:, tokens], axis=1) + 1
                new_head_probs = parse_probs[cycle, new_heads] / old_head_probs
                # Select the most probable change
                change = np.argmax(new_head_probs)
                changed_cycle = cycle[change]
                old_head = old_heads[change]
                new_head = new_heads[change]
                # Make the change
                parse_preds[changed_cycle] = new_head
                tarjan.edges[new_head].add(changed_cycle)
                tarjan.edges[old_head].remove(changed_cycle)
        return parse_preds
    else:
        # block and pad heads
        parse_probs = parse_probs * tokens_to_keep
        parse_preds = np.argmax(parse_probs, axis=1)
        return parse_preds


def chu_liu_edmonds(parse_probs, length):
    tree = decode_mst(parse_probs.T, length, False)[0]
    tree[0] = 0
    return tree


def unique_root(parse_probs, tokens_to_keep: np.ndarray, length):
    I = np.eye(len(tokens_to_keep))
    # block loops and pad heads
    if tokens_to_keep.ndim == 1:
        tokens_to_keep = np.expand_dims(tokens_to_keep, -1)
    parse_probs = parse_probs * tokens_to_keep * (1 - I)
    parse_preds = np.argmax(parse_probs, axis=1)
    tokens = np.arange(1, length)
    roots = np.where(parse_preds[tokens] == 0)[0] + 1
    # ensure at least one root
    if len(roots) < 1:
        # The current root probabilities
        root_probs = parse_probs[tokens, 0]
        # The current head probabilities
        old_head_probs = parse_probs[tokens, parse_preds[tokens]]
        # Get new potential root probabilities
        new_root_probs = root_probs / old_head_probs
        # Select the most probable root
        new_root = tokens[np.argmax(new_root_probs)]
        # Make the change
        parse_preds[new_root] = 0
    # ensure at most one root
    elif len(roots) > 1:
        # The probabilities of the current heads
        root_probs = parse_probs[roots, 0]
        # Set the probability of depending on the root zero
        parse_probs[roots, 0] = 0
        # Get new potential heads and their probabilities
        new_heads = np.argmax(parse_probs[roots][:, tokens], axis=1) + 1
        new_head_probs = parse_probs[roots, new_heads] / root_probs
        # Select the most probable root
        new_root = roots[np.argmin(new_head_probs)]
        # Make the change
        parse_preds[roots] = new_heads
        parse_preds[new_root] = 0
    return parse_preds, parse_probs, tokens


def dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path + [next_state]))


def mst_then_greedy(arc_scores, rel_scores, mask, root_rel_idx, rel_idx=None):
    from scipy.special import softmax
    from scipy.special import expit as sigmoid
    length = sum(mask) + 1
    mask = mask[:length]
    arc_scores = arc_scores[:length, :length]
    arc_pred = arc_scores > 0
    arc_probs = sigmoid(arc_scores)
    rel_scores = rel_scores[:length, :length, :]
    rel_probs = softmax(rel_scores, -1)
    if not any(arc_pred[:, 0][1:]):  # no root
        root = np.argmax(rel_probs[1:, 0, root_rel_idx]) + 1
        arc_probs[root, 0] = 1
    parse_preds, parse_probs, tokens = unique_root(arc_probs, mask, length)
    root = adjust_root_score(arc_scores, parse_preds, root_rel_idx, rel_scores)
    tree = chu_liu_edmonds(arc_scores, length)
    if rel_idx is not None:  # Unknown DEPREL label: 'ref'
        rel_scores[np.arange(len(tree)), tree, rel_idx] = -float('inf')
    return tree, add_secondary_arcs_by_scores(arc_scores, rel_scores, tree, root_rel_idx)


def adjust_root_score(arc_scores, parse_preds, root_rel_idx, rel_scores=None):
    root = np.where(parse_preds[1:] == 0)[0] + 1
    arc_scores[:, 0] = min(np.min(arc_scores), -1000)
    arc_scores[root, 0] = max(np.max(arc_scores), 1000)
    if rel_scores is not None:
        rel_scores[:, :, root_rel_idx] = -float('inf')
        rel_scores[root, 0, root_rel_idx] = float('inf')
    return root


def add_secondary_arcs_by_scores(arc_scores, rel_scores, tree, root_rel_idx, arc_preds=None):
    if not isinstance(tree, np.ndarray):
        tree = np.array(tree)
    if arc_preds is None:
        arc_preds = arc_scores > 0
    rel_pred = np.argmax(rel_scores, axis=-1)

    return add_secondary_arcs_by_preds(arc_scores, arc_preds, rel_pred, tree, root_rel_idx)


def add_secondary_arcs_by_preds(arc_scores, arc_preds, rel_preds, tree, root_rel_idx=None):
    dh = np.argwhere(arc_preds)
    sdh = sorted([(arc_scores[x[0], x[1]], list(x)) for x in dh], reverse=True)
    graph = [[] for _ in range(len(tree))]
    for d, h in enumerate(tree):
        if d:
            graph[h].append(d)
    for s, (d, h) in sdh:
        if not d or not h or d in graph[h]:
            continue
        try:
            path = next(dfs(graph, d, h))
        except StopIteration:
            # no path from d to h
            graph[h].append(d)
    parse_graph = [[] for _ in range(len(tree))]
    num_root = 0
    for h in range(len(tree)):
        for d in graph[h]:
            rel = rel_preds[d, h]
            if h == 0 and root_rel_idx is not None:
                rel = root_rel_idx
                assert num_root == 0
                num_root += 1
            parse_graph[d].append((h, rel))
        parse_graph[d] = sorted(parse_graph[d])
    return parse_graph


def adjust_root_score_then_add_secondary_arcs(arc_scores, rel_scores, tree, root_rel_idx):
    if len(arc_scores) != tree:
        arc_scores = arc_scores[:len(tree), :len(tree)]
        rel_scores = rel_scores[:len(tree), :len(tree), :]
    parse_preds = arc_scores > 0
    # adjust_root_score(arc_scores, parse_preds, rel_scores)
    parse_preds[:, 0] = False  # set heads to False
    rel_scores[:, :, root_rel_idx] = -float('inf')
    return add_secondary_arcs_by_scores(arc_scores, rel_scores, tree, root_rel_idx, parse_preds)
