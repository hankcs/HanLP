# MIT License
#
# Copyright (c) 2020 Yu Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch

from hanlp_common.conll import isprojective


def kmeans(x, k, max_it=32):
    r"""
    KMeans algorithm for clustering the sentences by length.

    Args:
        x (list[int]):
            The list of sentence lengths.
        k (int):
            The number of clusters.
            This is an approximate value. The final number of clusters can be less or equal to `k`.
        max_it (int):
            Maximum number of iterations.
            If centroids does not converge after several iterations, the algorithm will be early stopped.

    Returns:
        list[float], list[list[int]]:
            The first list contains average lengths of sentences in each cluster.
            The second is the list of clusters holding indices of data points.

    Examples:
        >>> x = torch.randint(10,20,(10,)).tolist()
        >>> x
        [15, 10, 17, 11, 18, 13, 17, 19, 18, 14]
        >>> centroids, clusters = kmeans(x, 3)
        >>> centroids
        [10.5, 14.0, 17.799999237060547]
        >>> clusters
        [[1, 3], [0, 5, 9], [2, 4, 6, 7, 8]]
    """

    # the number of clusters must not be greater than the number of datapoints
    x, k = torch.tensor(x, dtype=torch.float), min(len(x), k)
    # collect unique datapoints
    d = x.unique()
    # initialize k centroids randomly
    c = d[torch.randperm(len(d))[:k]]
    # assign each datapoint to the cluster with the closest centroid
    dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)

    for _ in range(max_it):
        # if an empty cluster is encountered,
        # choose the farthest datapoint from the biggest cluster and move that the empty one
        mask = torch.arange(k).unsqueeze(-1).eq(y)
        none = torch.where(~mask.any(-1))[0].tolist()
        while len(none) > 0:
            for i in none:
                # the biggest cluster
                b = torch.where(mask[mask.sum(-1).argmax()])[0]
                # the datapoint farthest from the centroid of cluster b
                f = dists[b].argmax()
                # update the assigned cluster of f
                y[b[f]] = i
                # re-calculate the mask
                mask = torch.arange(k).unsqueeze(-1).eq(y)
            none = torch.where(~mask.any(-1))[0].tolist()
        # update the centroids
        c, old = (x * mask).sum(-1) / mask.sum(-1), c
        # re-assign all datapoints to clusters
        dists, y = torch.abs_(x.unsqueeze(-1) - c).min(-1)
        # stop iteration early if the centroids converge
        if c.equal(old):
            break
    # assign all datapoints to the new-generated clusters
    # the empty ones are discarded
    assigned = y.unique().tolist()
    # get the centroids of the assigned clusters
    centroids = c[assigned].tolist()
    # map all values of datapoints to buckets
    clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]

    return centroids, clusters


def eisner(scores, mask):
    r"""
    First-order Eisner algorithm for projective decoding.

    References:
        - Ryan McDonald, Koby Crammer and Fernando Pereira. 2005.
          `Online Large-Margin Training of Dependency Parsers`_.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all dependent-head pairs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting projective parse trees.

    Examples:
        >>> scores = torch.tensor([[[-13.5026, -18.3700, -13.0033, -16.6809],
                                    [-36.5235, -28.6344, -28.4696, -31.6750],
                                    [ -2.9084,  -7.4825,  -1.4861,  -6.8709],
                                    [-29.4880, -27.6905, -26.1498, -27.0233]]])
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> eisner(scores, mask)
        tensor([[0, 2, 0, 2]])

    .. _Online Large-Margin Training of Dependency Parsers:
        https://www.aclweb.org/anthology/P05-1012/
    """

    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    p_i = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = scores.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        il = ir = ilr.permute(2, 0, 1)
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span + scores.diagonal(-w))
        p_i.diagonal(-w).copy_(il_path + starts)
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span + scores.diagonal(w))
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    def backtrack(p_i, p_c, heads, i, j, complete):
        if i == j:
            return
        if complete:
            r = p_c[i, j]
            backtrack(p_i, p_c, heads, i, r, False)
            backtrack(p_i, p_c, heads, r, j, True)
        else:
            r, heads[j] = p_i[i, j], i
            i, j = sorted((i, j))
            backtrack(p_i, p_c, heads, i, r, True)
            backtrack(p_i, p_c, heads, j, r + 1, True)

    preds = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_zeros(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        preds.append(heads.to(mask.device))

    return pad(preds, total_length=seq_len).to(mask.device)


def backtrack(p_i, p_c, heads, i, j, complete):
    if i == j:
        return
    if complete:
        r = p_c[i, j]
        backtrack(p_i, p_c, heads, i, r, False)
        backtrack(p_i, p_c, heads, r, j, True)
    else:
        r, heads[j] = p_i[i, j], i
        i, j = sorted((i, j))
        backtrack(p_i, p_c, heads, i, r, True)
        backtrack(p_i, p_c, heads, j, r + 1, True)


def stripe(x, n, w, offset=(0, 0), dim=1):
    """r'''Returns a diagonal stripe of the tensor.

    Args:
      x: Tensor
      n: int
      w: int
      offset: tuple (Default value = (0)
      dim: int (Default value = 1)
      Example: 
      0): 

    Returns:

    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    """
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def cky(scores, mask):
    r"""
    The implementation of `Cocke-Kasami-Younger`_ (CKY) algorithm to parse constituency trees.

    References:
        - Yu Zhang, Houquan Zhou and Zhenghua Li. 2020.
          `Fast and Accurate Neural CRF Constituency Parsing`_.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all candidate constituents.
        mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
            The mask to avoid parsing over padding tokens.
            For each square matrix in a batch, the positions except upper triangular part should be masked out.

    Returns:
        Sequences of factorized predicted bracketed trees that are traversed in pre-order.

    Examples:
        >>> scores = torch.tensor([[[ 2.5659,  1.4253, -2.5272,  3.3011],
                                    [ 1.3687, -0.5869,  1.0011,  3.3020],
                                    [ 1.2297,  0.4862,  1.1975,  2.5387],
                                    [-0.0511, -1.2541, -0.7577,  0.2659]]])
        >>> mask = torch.tensor([[[False,  True,  True,  True],
                                  [False, False,  True,  True],
                                  [False, False, False,  True],
                                  [False, False, False, False]]])
        >>> cky(scores, mask)
        [[(0, 3), (0, 1), (1, 3), (1, 2), (2, 3)]]

    .. _Cocke-Kasami-Younger:
        https://en.wikipedia.org/wiki/CYK_algorithm
    .. _Fast and Accurate Neural CRF Constituency Parsing:
        https://www.ijcai.org/Proceedings/2020/560/
    """

    lens = mask[:, 0].sum(-1)
    scores = scores.permute(1, 2, 0)
    seq_len, seq_len, batch_size = scores.shape
    s = scores.new_zeros(seq_len, seq_len, batch_size)
    p = scores.new_zeros(seq_len, seq_len, batch_size).long()

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p.new_tensor(range(n)).unsqueeze(0)

        if w == 1:
            s.diagonal(w).copy_(scores.diagonal(w))
            continue
        # [n, w, batch_size]
        s_span = stripe(s, n, w - 1, (0, 1)) + stripe(s, n, w - 1, (1, w), 0)
        # [batch_size, n, w]
        s_span = s_span.permute(2, 0, 1)
        # [batch_size, n]
        s_span, p_span = s_span.max(-1)
        s.diagonal(w).copy_(s_span + scores.diagonal(w))
        p.diagonal(w).copy_(p_span + starts + 1)

    def backtrack(p, i, j):
        if j == i + 1:
            return [(i, j)]
        split = p[i][j]
        ltree = backtrack(p, i, split)
        rtree = backtrack(p, split, j)
        return [(i, j)] + ltree + rtree

    p = p.permute(2, 0, 1).tolist()
    trees = [backtrack(p[i], 0, length) if length else [] for i, length in enumerate(lens.tolist())]

    return trees


def istree(sequence, proj=False, multiroot=False):
    r"""
    Checks if the arcs form an valid dependency tree.

    Args:
        sequence (list[int]):
            A list of head indices.
        proj (bool):
            If ``True``, requires the tree to be projective. Default: ``False``.
        multiroot (bool):
            If ``False``, requires the tree to contain only a single root. Default: ``True``.

    Returns:
        ``True`` if the arcs form an valid tree, ``False`` otherwise.

    Examples:
        >>> istree([3, 0, 0, 3], multiroot=True)
        True
        >>> istree([3, 0, 0, 3], proj=True)
        False
    """

    if proj and not isprojective(sequence):
        return False
    n_roots = sum(head == 0 for head in sequence)
    if n_roots == 0:
        return False
    if not multiroot and n_roots > 1:
        return False
    if any(i == head for i, head in enumerate(sequence, 1)):
        return False
    return next(tarjan(sequence), None) is None


def tarjan(sequence):
    r"""
    Tarjan algorithm for finding Strongly Connected Components (SCCs) of a graph.

    Args:
        sequence (list):
            List of head indices.

    Yields:
        A list of indices that make up a SCC. All self-loops are ignored.

    Examples:
        >>> next(tarjan([2, 5, 0, 3, 1]))  # (1 -> 5 -> 2 -> 1) is a cycle
        [2, 5, 1]
    """

    sequence = [-1] + sequence
    # record the search order, i.e., the timestep
    dfn = [-1] * len(sequence)
    # record the the smallest timestep in a SCC
    low = [-1] * len(sequence)
    # push the visited into the stack
    stack, onstack = [], [False] * len(sequence)

    def connect(i, timestep):
        dfn[i] = low[i] = timestep[0]
        timestep[0] += 1
        stack.append(i)
        onstack[i] = True

        for j, head in enumerate(sequence):
            if head != i:
                continue
            if dfn[j] == -1:
                yield from connect(j, timestep)
                low[i] = min(low[i], low[j])
            elif onstack[j]:
                low[i] = min(low[i], dfn[j])

        # a SCC is completed
        if low[i] == dfn[i]:
            cycle = [stack.pop()]
            while cycle[-1] != i:
                onstack[cycle[-1]] = False
                cycle.append(stack.pop())
            onstack[i] = False
            # ignore the self-loop
            if len(cycle) > 1:
                yield cycle

    timestep = [0]
    for i in range(len(sequence)):
        if dfn[i] == -1:
            yield from connect(i, timestep)


def chuliu_edmonds(s):
    r"""
    ChuLiu/Edmonds algorithm for non-projective decoding.

    Some code is borrowed from `tdozat's implementation`_.
    Descriptions of notations and formulas can be found in
    `Non-projective Dependency Parsing using Spanning Tree Algorithms`_.

    Notes:
        The algorithm does not guarantee to parse a single-root tree.

    References:
        - Ryan McDonald, Fernando Pereira, Kiril Ribarov and Jan Hajic. 2005.
          `Non-projective Dependency Parsing using Spanning Tree Algorithms`_.

    Args:
        s (~torch.Tensor): ``[seq_len, seq_len]``.
            Scores of all dependent-head pairs.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[seq_len]`` for the resulting non-projective parse tree.

    .. _tdozat's implementation:
        https://github.com/tdozat/Parser-v3
    .. _Non-projective Dependency Parsing using Spanning Tree Algorithms:
        https://www.aclweb.org/anthology/H05-1066/
    """

    s[0, 1:] = float('-inf')
    # prevent self-loops
    s.diagonal()[1:].fill_(float('-inf'))
    # select heads with highest scores
    tree = s.argmax(-1)
    # return the cycle finded by tarjan algorithm lazily
    cycle = next(tarjan(tree.tolist()[1:]), None)
    # if the tree has no cycles, then it is a MST
    if not cycle:
        return tree
    # indices of cycle in the original tree
    cycle = torch.tensor(cycle)
    # indices of noncycle in the original tree
    noncycle = torch.ones(len(s)).index_fill_(0, cycle, 0)
    noncycle = torch.where(noncycle.gt(0))[0]

    def contract(s):
        # heads of cycle in original tree
        cycle_heads = tree[cycle]
        # scores of cycle in original tree
        s_cycle = s[cycle, cycle_heads]

        # calculate the scores of cycle's potential dependents
        # s(c->x) = max(s(x'->x)), x in noncycle and x' in cycle
        s_dep = s[noncycle][:, cycle]
        # find the best cycle head for each noncycle dependent
        deps = s_dep.argmax(1)
        # calculate the scores of cycle's potential heads
        # s(x->c) = max(s(x'->x) - s(a(x')->x') + s(cycle)), x in noncycle and x' in cycle
        #                                                    a(v) is the predecessor of v in cycle
        #                                                    s(cycle) = sum(s(a(v)->v))
        s_head = s[cycle][:, noncycle] - s_cycle.view(-1, 1) + s_cycle.sum()
        # find the best noncycle head for each cycle dependent
        heads = s_head.argmax(0)

        contracted = torch.cat((noncycle, torch.tensor([-1])))
        # calculate the scores of contracted graph
        s = s[contracted][:, contracted]
        # set the contracted graph scores of cycle's potential dependents
        s[:-1, -1] = s_dep[range(len(deps)), deps]
        # set the contracted graph scores of cycle's potential heads
        s[-1, :-1] = s_head[heads, range(len(heads))]

        return s, heads, deps

    # keep track of the endpoints of the edges into and out of cycle for reconstruction later
    s, heads, deps = contract(s)

    # y is the contracted tree
    y = chuliu_edmonds(s)
    # exclude head of cycle from y
    y, cycle_head = y[:-1], y[-1]

    # fix the subtree with no heads coming from the cycle
    # len(y) denotes heads coming from the cycle
    subtree = y < len(y)
    # add the nodes to the new tree
    tree[noncycle[subtree]] = noncycle[y[subtree]]
    # fix the subtree with heads coming from the cycle
    subtree = ~subtree
    # add the nodes to the tree
    tree[noncycle[subtree]] = cycle[deps[subtree]]
    # fix the root of the cycle
    cycle_root = heads[cycle_head]
    # break the cycle and add the root of the cycle to the tree
    tree[cycle[cycle_root]] = noncycle[cycle_head]

    return tree


def mst(scores, mask, multiroot=False):
    r"""
    MST algorithm for decoding non-pojective trees.
    This is a wrapper for ChuLiu/Edmonds algorithm.

    The algorithm first runs ChuLiu/Edmonds to parse a tree and then have a check of multi-roots,
    If ``multiroot=True`` and there indeed exist multi-roots, the algorithm seeks to find
    best single-root trees by iterating all possible single-root trees parsed by ChuLiu/Edmonds.
    Otherwise the resulting trees are directly taken as the final outputs.

    Args:
        scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all dependent-head pairs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.
        muliroot (bool):
            Ensures to parse a single-root tree If ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting non-projective parse trees.

    Examples:
        >>> scores = torch.tensor([[[-11.9436, -13.1464,  -6.4789, -13.8917],
                                    [-60.6957, -60.2866, -48.6457, -63.8125],
                                    [-38.1747, -49.9296, -45.2733, -49.5571],
                                    [-19.7504, -23.9066,  -9.9139, -16.2088]]])
        >>> scores[:, 0, 1:] = float('-inf')
        >>> scores.diagonal(0, 1, 2)[1:].fill_(float('-inf'))
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> mst(scores, mask)
        tensor([[0, 2, 0, 2]])
    """

    batch_size, seq_len, _ = scores.shape
    scores = scores.detach().cpu().unbind()

    preds = []
    for i, length in enumerate(mask.sum(1).tolist()):
        s = scores[i][:length + 1, :length + 1]
        tree = chuliu_edmonds(s)
        roots = torch.where(tree[1:].eq(0))[0] + 1
        if not multiroot and len(roots) > 1:
            s_root = s[:, 0]
            s_best = float('-inf')
            s = s.index_fill(1, torch.tensor(0), float('-inf'))
            for root in roots:
                s[:, 0] = float('-inf')
                s[root, 0] = s_root[root]
                t = chuliu_edmonds(s)
                s_tree = s[1:].gather(1, t[1:].unsqueeze(-1)).sum()
                if s_tree > s_best:
                    s_best, tree = s_tree, t
        preds.append(tree)

    return pad(preds, total_length=seq_len).to(mask.device)


def eisner2o(scores, mask):
    r"""
    Second-order Eisner algorithm for projective decoding.
    This is an extension of the first-order one that further incorporates sibling scores into tree scoring.

    References:
        - Ryan McDonald and Fernando Pereira. 2006.
          `Online Learning of Approximate Dependency Parsing Algorithms`_.

    Args:
        scores (~torch.Tensor, ~torch.Tensor):
            A tuple of two tensors representing the first-order and second-order scores repectively.
            The first (``[batch_size, seq_len, seq_len]``) holds scores of all dependent-head pairs.
            The second (``[batch_size, seq_len, seq_len, seq_len]``) holds scores of all dependent-head-sibling triples.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask to avoid parsing over padding tokens.
            The first column serving as pseudo words for roots should be ``False``.

    Returns:
        ~torch.Tensor:
            A tensor with shape ``[batch_size, seq_len]`` for the resulting projective parse trees.

    Examples:
        >>> s_arc = torch.tensor([[[ -2.8092,  -7.9104,  -0.9414,  -5.4360],
                                   [-10.3494,  -7.9298,  -3.6929,  -7.3985],
                                   [  1.1815,  -3.8291,   2.3166,  -2.7183],
                                   [ -3.9776,  -3.9063,  -1.6762,  -3.1861]]])
        >>> s_sib = torch.tensor([[[[ 0.4719,  0.4154,  1.1333,  0.6946],
                                    [ 1.1252,  1.3043,  2.1128,  1.4621],
                                    [ 0.5974,  0.5635,  1.0115,  0.7550],
                                    [ 1.1174,  1.3794,  2.2567,  1.4043]],
                                   [[-2.1480, -4.1830, -2.5519, -1.8020],
                                    [-1.2496, -1.7859, -0.0665, -0.4938],
                                    [-2.6171, -4.0142, -2.9428, -2.2121],
                                    [-0.5166, -1.0925,  0.5190,  0.1371]],
                                   [[ 0.5827, -1.2499, -0.0648, -0.0497],
                                    [ 1.4695,  0.3522,  1.5614,  1.0236],
                                    [ 0.4647, -0.7996, -0.3801,  0.0046],
                                    [ 1.5611,  0.3875,  1.8285,  1.0766]],
                                   [[-1.3053, -2.9423, -1.5779, -1.2142],
                                    [-0.1908, -0.9699,  0.3085,  0.1061],
                                    [-1.6783, -2.8199, -1.8853, -1.5653],
                                    [ 0.3629, -0.3488,  0.9011,  0.5674]]]])
        >>> mask = torch.tensor([[False,  True,  True,  True]])
        >>> eisner2o((s_arc, s_sib), mask)
        tensor([[0, 2, 0, 2]])

    .. _Online Learning of Approximate Dependency Parsing Algorithms:
        https://www.aclweb.org/anthology/E06-1011/
    """

    # the end position of each sentence in a batch
    lens = mask.sum(1)
    s_arc, s_sib = scores
    batch_size, seq_len, _ = s_arc.shape
    # [seq_len, seq_len, batch_size]
    s_arc = s_arc.permute(2, 1, 0)
    # [seq_len, seq_len, seq_len, batch_size]
    s_sib = s_sib.permute(2, 1, 3, 0)
    s_i = torch.full_like(s_arc, float('-inf'))
    s_s = torch.full_like(s_arc, float('-inf'))
    s_c = torch.full_like(s_arc, float('-inf'))
    p_i = s_arc.new_zeros(seq_len, seq_len, batch_size).long()
    p_s = s_arc.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = s_arc.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        # n denotes the number of spans to iterate,
        # from span (0, w) to span (n, n+w) given width w
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # I(j->i) = max(I(j->r) + S(j->r, i)), i < r < j |
        #               C(j->j) + C(i->j-1))
        #           + s(j->i)
        # [n, w, batch_size]
        il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
        il += stripe(s_sib[range(w, n + w), range(n)], n, w, (0, 1))
        # [n, 1, batch_size]
        il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
        # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
        il[:, -1] = il0.index_fill_(0, lens.new_tensor(0), 0).squeeze(1)
        il_span, il_path = il.permute(2, 0, 1).max(-1)
        s_i.diagonal(-w).copy_(il_span + s_arc.diagonal(-w))
        p_i.diagonal(-w).copy_(il_path + starts + 1)
        # I(i->j) = max(I(i->r) + S(i->r, j), i < r < j |
        #               C(i->i) + C(j->i+1))
        #           + s(i->j)
        # [n, w, batch_size]
        ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
        ir += stripe(s_sib[range(n), range(w, n + w)], n, w)
        ir[0] = float('-inf')
        # [n, 1, batch_size]
        ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
        ir[:, 0] = ir0.squeeze(1)
        ir_span, ir_path = ir.permute(2, 0, 1).max(-1)
        s_i.diagonal(w).copy_(ir_span + s_arc.diagonal(w))
        p_i.diagonal(w).copy_(ir_path + starts)

        # [n, w, batch_size]
        slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        slr_span, slr_path = slr.permute(2, 0, 1).max(-1)
        # S(j, i) = max(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(-w).copy_(slr_span)
        p_s.diagonal(-w).copy_(slr_path + starts)
        # S(i, j) = max(C(i->r) + C(j->r+1)), i <= r < j
        s_s.diagonal(w).copy_(slr_span)
        p_s.diagonal(w).copy_(slr_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        # disable multi words to modify the root
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    def backtrack(p_i, p_s, p_c, heads, i, j, flag):
        if i == j:
            return
        if flag == 'c':
            r = p_c[i, j]
            backtrack(p_i, p_s, p_c, heads, i, r, 'i')
            backtrack(p_i, p_s, p_c, heads, r, j, 'c')
        elif flag == 's':
            r = p_s[i, j]
            i, j = sorted((i, j))
            backtrack(p_i, p_s, p_c, heads, i, r, 'c')
            backtrack(p_i, p_s, p_c, heads, j, r + 1, 'c')
        elif flag == 'i':
            r, heads[j] = p_i[i, j], i
            if r == i:
                r = i + 1 if i < j else i - 1
                backtrack(p_i, p_s, p_c, heads, j, r, 'c')
            else:
                backtrack(p_i, p_s, p_c, heads, i, r, 'i')
                backtrack(p_i, p_s, p_c, heads, r, j, 's')

    preds = []
    p_i = p_i.permute(2, 0, 1).cpu()
    p_s = p_s.permute(2, 0, 1).cpu()
    p_c = p_c.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_zeros(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_s[i], p_c[i], heads, 0, length, 'c')
        preds.append(heads.to(mask.device))

    return pad(preds, total_length=seq_len).to(mask.device)


def pad(tensors, padding_value=0, total_length=None):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def decode_dep(s_arc, mask, tree=False, proj=False):
    r"""
    Args:
        s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
            Scores of all possible arcs.
        mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
            The mask for covering the unpadded tokens.
        tree (bool):
            If ``True``, ensures to output well-formed trees. Default: ``False``.
        proj (bool):
            If ``True``, ensures to output projective trees. Default: ``False``.

    Returns:
        ~torch.Tensor, ~torch.Tensor:
            Predicted arcs and labels of shape ``[batch_size, seq_len]``.
    """

    lens = mask.sum(1)
    arc_preds = s_arc.argmax(-1)
    bad = [not istree(seq[1:i + 1], proj) for i, seq in zip(lens.tolist(), arc_preds.tolist())]
    if tree and any(bad):
        if proj:
            alg = eisner
        else:
            alg = mst
            s_arc.diagonal(0, 1, 2)[1:].fill_(float('-inf'))
        arc_preds[bad] = alg(s_arc[bad], mask[bad])

    return arc_preds
