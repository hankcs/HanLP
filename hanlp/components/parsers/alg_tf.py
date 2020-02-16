# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-26 19:49
# Ported from the PyTorch implementation https://github.com/zysite/biaffine-parser
from typing import List
import numpy as np
import tensorflow as tf
from collections import defaultdict


def nonzero(t: tf.Tensor) -> tf.Tensor:
    return tf.where(t > 0)


def view(t: tf.Tensor, *dims) -> tf.Tensor:
    return tf.reshape(t, dims)


def arange(n: int) -> tf.Tensor:
    return tf.range(n)


def randperm(n: int) -> tf.Tensor:
    return tf.random.shuffle(arange(n))


def tolist(t: tf.Tensor) -> List:
    if isinstance(t, tf.Tensor):
        t = t.numpy()
    return t.tolist()


def kmeans(x, k, seed=None):
    """See https://github.com/zysite/biaffine-parser/blob/master/parser/utils/alg.py#L7

    Args:
      x(list): Lengths of sentences
      k(int): 
      seed:  (Default value = None)

    Returns:

    
    """
    x = tf.constant(x, dtype=tf.float32)
    # count the frequency of each datapoint
    d, indices, f = tf.unique_with_counts(x, tf.int32)
    f = tf.cast(f, tf.float32)
    # calculate the sum of the values of the same datapoints
    total = d * f
    # initialize k centroids randomly
    c, old = tf.random.shuffle(d, seed)[:k], None
    # assign labels to each datapoint based on centroids
    dists = tf.abs(tf.expand_dims(d, -1) - c)
    y = tf.argmin(dists, axis=-1, output_type=tf.int32)
    dists = tf.gather_nd(dists, tf.transpose(tf.stack([tf.range(tf.shape(dists)[0], dtype=tf.int32), y])))
    # make sure number of datapoints is greater than that of clusters
    assert len(d) >= k, f"unable to assign {len(d)} datapoints to {k} clusters"

    while old is None or not tf.reduce_all(c == old):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster
        # and move that the empty one
        for i in range(k):
            if not tf.reduce_any(y == i):
                mask = tf.cast(y == tf.expand_dims(tf.range(k, dtype=tf.int32), -1), tf.float32)
                lens = tf.reduce_sum(mask, axis=-1)
                biggest = view(nonzero(mask[tf.argmax(lens)]), -1)
                farthest = tf.argmax(tf.gather(dists, biggest))
                tf.tensor_scatter_nd_update(y, tf.expand_dims(tf.expand_dims(biggest[farthest], -1), -1), [i])
        mask = tf.cast(y == tf.expand_dims(tf.range(k, dtype=tf.int32), -1), tf.float32)
        # update the centroids
        c, old = tf.cast(tf.reduce_sum(total * mask, axis=-1), tf.float32) / tf.cast(tf.reduce_sum(f * mask, axis=-1),
                                                                                     tf.float32), c
        # re-assign all datapoints to clusters
        dists = tf.abs(tf.expand_dims(d, -1) - c)
        y = tf.argmin(dists, axis=-1, output_type=tf.int32)
        dists = tf.gather_nd(dists, tf.transpose(tf.stack([tf.range(tf.shape(dists)[0], dtype=tf.int32), y])))
    # assign all datapoints to the new-generated clusters
    # without considering the empty ones
    y, (assigned, _) = tf.gather(y, indices), tf.unique(y)
    # get the centroids of the assigned clusters
    centroids = tf.gather(c, assigned).numpy().tolist()
    # map all values of datapoints to buckets
    clusters = [tf.squeeze(tf.where(y == i), axis=-1).numpy().tolist() for i in assigned]

    return centroids, clusters


# ***************************************************************
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
        I = np.eye(len(tokens_to_keep))
        # block loops and pad heads
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


def rel_argmax(rel_probs, length, root, ensure_tree=True):
    """Fix the relation prediction by heuristic rules

    Args:
      rel_probs(NDArray): seq_len x rel_size
      length: real sentence length
      ensure_tree:  (Default value = True)
      root: 

    Returns:

    
    """
    if ensure_tree:
        tokens = np.arange(1, length)
        rel_preds = np.argmax(rel_probs, axis=1)
        roots = np.where(rel_preds[tokens] == root)[0] + 1
        if len(roots) < 1:
            rel_preds[1 + np.argmax(rel_probs[tokens, root])] = root
        elif len(roots) > 1:
            root_probs = rel_probs[roots, root]
            rel_probs[roots, root] = 0
            new_rel_preds = np.argmax(rel_probs[roots], axis=1)
            new_rel_probs = rel_probs[roots, new_rel_preds] / root_probs
            new_root = roots[np.argmin(new_rel_probs)]
            rel_preds[roots] = new_rel_preds
            rel_preds[new_root] = root
        return rel_preds
    else:
        rel_preds = np.argmax(rel_probs, axis=1)
        return rel_preds
