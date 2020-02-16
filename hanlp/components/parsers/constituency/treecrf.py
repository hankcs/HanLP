# -*- coding:utf-8 -*-
# Adopted from https://github.com/yzhangcs/parser
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
import torch.autograd as autograd
import torch.nn as nn

from hanlp.components.parsers.alg import stripe, istree, eisner, mst, eisner2o


class CRFConstituency(nn.Module):
    r"""
    TreeCRF for calculating partition functions and marginals in :math:`O(n^3)` for constituency trees.

    References:
        - Yu Zhang, houquan Zhou and Zhenghua Li. 2020.
          `Fast and Accurate Neural CRF Constituency Parsing`_.

    .. _Fast and Accurate Neural CRF Constituency Parsing:
        https://www.ijcai.org/Proceedings/2020/560/
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible constituents.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid parsing over padding tokens.
                For each square matrix in a batch, the positions except upper triangular part should be masked out.
            target (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard constituents. ``True`` if a constituent exists. Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        training = scores.requires_grad
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside(scores.requires_grad_(), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        probs = scores
        if mbr:
            probs, = autograd.grad(logZ, scores, retain_graph=training)
        if target is None:
            return probs
        loss = (logZ - scores[mask & target].sum()) / mask[:, 0].sum()

        return loss, probs

    def inside(self, scores, mask):
        lens = mask[:, 0].sum(-1)
        batch_size, seq_len, _ = scores.shape
        # [seq_len, seq_len, batch_size]
        scores, mask = scores.permute(1, 2, 0), mask.permute(1, 2, 0)
        s = torch.full_like(scores, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w

            if w == 1:
                s.diagonal(w).copy_(scores.diagonal(w))
                continue
            # [n, w, batch_size]
            s_s = stripe(s, n, w - 1, (0, 1)) + stripe(s, n, w - 1, (1, w), 0)
            # [batch_size, n, w]
            s_s = s_s.permute(2, 0, 1)
            if s_s.requires_grad:
                s_s.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_s = s_s.logsumexp(-1)
            s.diagonal(w).copy_(s_s + scores.diagonal(w))

        return s[0].gather(0, lens.unsqueeze(0)).sum()


class CRF2oDependency(nn.Module):
    r"""
    Second-order TreeCRF for calculating partition functions and marginals in :math:`O(n^3)` for projective dependency trees.

    References:
        - Yu Zhang, Zhenghua Li and Min Zhang. 2020.
          `Efficient Second-Order TreeCRF for Neural Dependency Parsing`_.

    .. _Efficient Second-Order TreeCRF for Neural Dependency Parsing:
        https://www.aclweb.org/anthology/2020.acl-main.302/
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, mbr=True, partial=False):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of two tensors `s_arc` and `s_sib`.
                `s_arc` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                Tensors of gold-standard dependent-head pairs and dependent-head-sibling triples.
                If partially annotated, the unannotated positions should be filled with -1.
                Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.
            partial (bool):
                ``True`` indicates that the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """

        s_arc, s_sib = scores
        training = s_arc.requires_grad
        batch_size, seq_len, _ = s_arc.shape
        # always enable the gradient computation of scores in order for the computation of marginals
        logZ = self.inside((s.requires_grad_() for s in scores), mask)
        # marginals are used for decoding, and can be computed by combining the inside pass and autograd mechanism
        probs = s_arc
        if mbr:
            probs, = autograd.grad(logZ, s_arc, retain_graph=training)

        if target is None:
            return probs
        arcs, sibs = target
        # the second inside process is needed if use partial annotation
        if partial:
            score = self.inside(scores, mask, arcs)
        else:
            arc_seq, sib_seq = arcs[mask], sibs[mask]
            arc_mask, sib_mask = mask, sib_seq.gt(0)
            sib_seq = sib_seq[sib_mask]
            s_sib = s_sib[mask][torch.arange(len(arc_seq)), arc_seq]
            s_arc = s_arc[arc_mask].gather(-1, arc_seq.unsqueeze(-1))
            s_sib = s_sib[sib_mask].gather(-1, sib_seq.unsqueeze(-1))
            score = s_arc.sum() + s_sib.sum()
        loss = (logZ - score) / mask.sum()

        return loss, probs

    def inside(self, scores, mask, cands=None):
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
        s_c.diagonal().fill_(0)

        # set the scores of arcs excluded by cands to -inf
        if cands is not None:
            mask = mask.index_fill(1, lens.new_tensor(0), 1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(-1)).permute(2, 1, 0)
            cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)
            cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
            cands = cands.permute(2, 1, 0) & mask
            s_arc = s_arc.masked_fill(~cands, float('-inf'))

        for w in range(1, seq_len):
            # n denotes the number of spans to iterate,
            # from span (0, w) to span (n, n+w) given width w
            n = seq_len - w
            # I(j->i) = logsum(exp(I(j->r) + S(j->r, i)) +, i < r < j
            #                  exp(C(j->j) + C(i->j-1)))
            #           + s(j->i)
            # [n, w, batch_size]
            il = stripe(s_i, n, w, (w, 1)) + stripe(s_s, n, w, (1, 0), 0)
            il += stripe(s_sib[range(w, n + w), range(n)], n, w, (0, 1))
            # [n, 1, batch_size]
            il0 = stripe(s_c, n, 1, (w, w)) + stripe(s_c, n, 1, (0, w - 1))
            # il0[0] are set to zeros since the scores of the complete spans starting from 0 are always -inf
            il[:, -1] = il0.index_fill_(0, lens.new_tensor(0), 0).squeeze(1)
            if il.requires_grad:
                il.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            il = il.permute(2, 0, 1).logsumexp(-1)
            s_i.diagonal(-w).copy_(il + s_arc.diagonal(-w))
            # I(i->j) = logsum(exp(I(i->r) + S(i->r, j)) +, i < r < j
            #                  exp(C(i->i) + C(j->i+1)))
            #           + s(i->j)
            # [n, w, batch_size]
            ir = stripe(s_i, n, w) + stripe(s_s, n, w, (0, w), 0)
            ir += stripe(s_sib[range(n), range(w, n + w)], n, w)
            ir[0] = float('-inf')
            # [n, 1, batch_size]
            ir0 = stripe(s_c, n, 1) + stripe(s_c, n, 1, (w, 1))
            ir[:, 0] = ir0.squeeze(1)
            if ir.requires_grad:
                ir.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            ir = ir.permute(2, 0, 1).logsumexp(-1)
            s_i.diagonal(w).copy_(ir + s_arc.diagonal(w))

            # [n, w, batch_size]
            slr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
            if slr.requires_grad:
                slr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            slr = slr.permute(2, 0, 1).logsumexp(-1)
            # S(j, i) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(-w).copy_(slr)
            # S(i, j) = logsumexp(C(i->r) + C(j->r+1)), i <= r < j
            s_s.diagonal(w).copy_(slr)

            # C(j->i) = logsumexp(C(r->i) + I(j->r)), i <= r < j
            cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
            cl.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(-w).copy_(cl.permute(2, 0, 1).logsumexp(-1))
            # C(i->j) = logsumexp(I(i->r) + C(r->j)), i < r <= j
            cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
            cr.register_hook(lambda x: x.masked_fill_(torch.isnan(x), 0))
            s_c.diagonal(w).copy_(cr.permute(2, 0, 1).logsumexp(-1))
            # disable multi words to modify the root
            s_c[0, w][lens.ne(w)] = float('-inf')

        return s_c[0].gather(0, lens.unsqueeze(0)).sum()

    def loss(self, s_arc, s_sib, s_rel, arcs, sibs, rels, mask, mbr=True, partial=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            arcs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard arcs.
            sibs (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard siblings.
            rels (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.
            partial (bool):
                ``True`` denotes the trees are partially annotated. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                original arc scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """

        scores, target = (s_arc, s_sib), (arcs, sibs)
        arc_loss, arc_probs = self.forward(scores, mask, target, mbr, partial)
        # -1 denotes un-annotated arcs
        if partial:
            mask = mask & arcs.ge(0)
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(rels)), arcs[mask]]
        rel_loss = self.criterion(s_rel, rels)
        loss = arc_loss + rel_loss
        return loss, arc_probs

    # def decode(self, s_arc, s_rel, mask, tree=False, proj=False, alg=None):
    #     r"""
    #     Args:
    #         s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
    #             Scores of all possible arcs.
    #         s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
    #             Scores of all possible labels on each arc.
    #         mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
    #             The mask for covering the unpadded tokens.
    #         tree (bool):
    #             If ``True``, ensures to output well-formed trees. Default: ``False``.
    #         proj (bool):
    #             If ``True``, ensures to output projective trees. Default: ``False``.
    # 
    #     Returns:
    #         ~torch.Tensor, ~torch.Tensor:
    #             Predicted arcs and labels of shape ``[batch_size, seq_len]``.
    #     """
    # 
    #     lens = mask.sum(1)
    #     arc_preds = s_arc.argmax(-1)
    #     if tree and not alg:
    #         bad = [not istree(seq[1:i + 1], proj)
    #                for i, seq in zip(lens.tolist(), arc_preds.tolist())]
    #         if any(bad):
    #             alg = eisner if proj else mst
    #             arc_preds[bad] = alg(s_arc[bad], mask[bad])
    #     rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
    # 
    #     return arc_preds, rel_preds
    def decode(self, s_arc, s_sib, s_rel, mask, tree=False, mbr=True, proj=False):
        r"""
        Args:
            s_arc (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible arcs.
            s_sib (~torch.Tensor): ``[batch_size, seq_len, seq_len, seq_len]``.
                Scores of all possible dependent-head-sibling triples.
            s_rel (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each arc.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.
            tree (bool):
                If ``True``, ensures to output well-formed trees. Default: ``False``.
            mbr (bool):
                If ``True``, performs MBR decoding. Default: ``True``.
            proj (bool):
                If ``True``, ensures to output projective trees. Default: ``False``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                Predicted arcs and labels of shape ``[batch_size, seq_len]``.
        """

        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        if tree:
            bad = [not istree(seq[1:i + 1], proj)
                   for i, seq in zip(lens.tolist(), arc_preds.tolist())]
            if any(bad):
                if proj and not mbr:
                    arc_preds = eisner2o((s_arc, s_sib), mask)
                else:
                    alg = eisner if proj else mst
                    arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds
