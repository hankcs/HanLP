# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-22 23:41
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
import pyximport
from hanlp.components.parsers.hpsg.trees import InternalTreebankNode, InternalParseNode

pyximport.install(setup_args={"include_dirs": np.get_include()})
from hanlp.components.parsers.hpsg import hpsg_decoder
from hanlp.components.parsers.hpsg import const_decoder
from hanlp.components.parsers.hpsg import trees
from alnlp.modules import util

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
ROOT = "<START>"
Sub_Head = "<H>"
No_Head = "<N>"

DTYPE = torch.bool

TAG_UNK = "UNK"

ROOT_TYPE = "<ROOT_TYPE>"

# Assumes that these control characters are not present in treebank text
CHAR_UNK = "\0"
CHAR_START_SENTENCE = "\1"
CHAR_START_WORD = "\2"
CHAR_STOP_WORD = "\3"
CHAR_STOP_SENTENCE = "\4"
CHAR_PAD = "\5"


def from_numpy(ndarray):
    return torch.from_numpy(ndarray)


class BatchIndices:
    """Batch indices container class (used to implement packed batches)"""

    def __init__(self, batch_idxs_torch):
        self.batch_idxs_torch = batch_idxs_torch
        self.batch_size = int(1 + batch_idxs_torch.max())
        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_torch.cpu().numpy(), [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))


#
class LockedDropoutFunction(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, batch_idxs=None, p=0.5, train=False, inplace=False):
        """Tokens in the same batch share the same dropout mask

        Args:
          ctx: 
          input: 
          batch_idxs:  (Default value = None)
          p:  (Default value = 0.5)
          train:  (Default value = False)
          inplace:  (Default value = False)

        Returns:

        
        """
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            if batch_idxs:
                ctx.noise = input.new().resize_(batch_idxs.batch_size, input.size(1))
                ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
            else:
                ctx.noise = input.new(input.size(0), 1, input.size(2))
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None, None
        else:
            return grad_output, None, None, None, None


#
class FeatureDropout(nn.Module):
    """Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.

    Args:

    Returns:

    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input, batch_idxs=None):
        return LockedDropoutFunction.apply(input, batch_idxs, self.p, self.training, self.inplace)


#
class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


#
class ScaledAttention(nn.Module):
    def __init__(self, hparams, attention_dropout=0.1):
        super(ScaledAttention, self).__init__()
        self.hparams = hparams
        self.temper = hparams.d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat]
        # k: [batch, slot, feat]
        # v: [batch, slot, feat]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn.transpose(1, 2)).transpose(1, 2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


# %%

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat] or (batch * d_l) x max_len x d_k
        # k: [batch, slot, feat] or (batch * d_l) x max_len x d_k
        # v: [batch, slot, feat] or (batch * d_l) x max_len x d_v
        # q in LAL is (batch * d_l) x 1 x d_k

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper  # (batch * d_l) x max_len x max_len
        # in LAL, gives: (batch * d_l) x 1 x max_len
        # attention weights from each word to each word, for each label
        # in best model (repeated q): attention weights from label (as vector weights) to each word

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        # Note that this makes the distribution not sum to 1. At some point it
        # may be worth researching whether this is the right way to apply
        # dropout to the attention.
        # Note that the t2t code also applies dropout in this manner
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # (batch * d_l) x max_len x d_v
        # in LAL, gives: (batch * d_l) x 1 x d_v

        return output, attn


#
class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""

    def __init__(self, hparams, n_head, d_model, d_k, d_v, residual_dropout=0.1, attention_dropout=0.1,
                 d_positional=None):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.hparams = hparams

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            self.w_qs1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_ks1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_vs1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_v // 2))

            self.w_qs2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_ks2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_vs2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_v // 2))

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            self.proj = nn.Linear(n_head * d_v, d_model, bias=False)
        else:
            self.proj1 = nn.Linear(n_head * (d_v // 2), self.d_content, bias=False)
            self.proj2 = nn.Linear(n_head * (d_v // 2), self.d_positional, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, qk_inp=None):
        v_inp_repeated = inp.repeat(self.n_head, 1).view(self.n_head, -1, inp.size(-1))  # n_head x len_inp x d_model
        if qk_inp is None:
            qk_inp_repeated = v_inp_repeated
        else:
            qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, -1, qk_inp.size(-1))

        if not self.partitioned:
            q_s = torch.bmm(qk_inp_repeated, self.w_qs)  # n_head x len_inp x d_k
            k_s = torch.bmm(qk_inp_repeated, self.w_ks)  # n_head x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs)  # n_head x len_inp x d_v
        else:
            q_s = torch.cat([
                torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_qs1),
                torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_qs2),
            ], -1)
            k_s = torch.cat([
                torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_ks1),
                torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_ks2),
            ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:, :, :self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:, :, self.d_content:], self.w_vs2),
            ], -1)
        return q_s, k_s, v_s

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs, mask):
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * B) x T x d
        # (along with masks for the attention and output)
        n_head = self.n_head
        d_k, d_v = self.d_k, self.d_v

        T = batch_idxs.max_len
        B = batch_idxs.batch_size
        q_padded = self.pad_seuence(q_s, batch_idxs)
        k_padded = self.pad_seuence(k_s, batch_idxs)
        v_padded = self.pad_seuence(v_s, batch_idxs)

        return (
            q_padded.view(-1, T, d_k),
            k_padded.view(-1, T, d_k),
            v_padded.view(-1, T, d_v),
            ~mask.unsqueeze(1).expand(B, T, T).repeat(n_head, 1, 1),
            mask.repeat(n_head, 1),
        )

    @staticmethod
    def pad_seuence(q_s, batch_idxs):
        q_padded = pad_sequence(torch.split(q_s.transpose(0, 1), batch_idxs.seq_lens_np.tolist()), True).transpose(0,
                                                                                                                   2).contiguous()
        return q_padded

    def combine_v(self, outputs):
        # Combine attention information from the different heads
        n_head = self.n_head
        outputs = outputs.view(n_head, -1, self.d_v)  # n_head x len_inp x d_kv

        if not self.partitioned:
            # Switch from n_head x len_inp x d_v to len_inp x (n_head * d_v)
            outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)

            # Project back to residual size
            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:, :, :d_v1]
            outputs2 = outputs[:, :, d_v1:]
            outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
            ], -1)

        return outputs

    def forward(self, inp, batch_idxs, qk_inp=None, batch=None, batched_inp=None, **kwargs):
        residual = inp
        mask = batch['mask']
        B, T = mask.size()

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, qk_inp=qk_inp)
        # n_head x len_inp x d_kv

        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs, mask)
        # (n_head * batch) x len_padded x d_kv
        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask,
        )
        outputs = outputs_padded[output_mask]
        # (n_head * len_inp) x d_kv
        outputs = self.combine_v(outputs)
        # len_inp x d_model

        outputs = self.residual_dropout(outputs, batch_idxs)

        return self.layer_norm(outputs + residual), attns_padded


#
class PositionwiseFeedForward(nn.Module):
    """A position-wise feed forward module.
    
    Projects to a higher-dimensional space before applying ReLU, then projects
    back.

    Args:

    Returns:

    """

    def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_ff)
        self.w_2 = nn.Linear(d_ff, d_hid)

        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x

        output = self.w_1(x)
        output = self.relu_dropout(self.relu(output), batch_idxs)
        output = self.w_2(output)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)


#
class PartitionedPositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_ff, d_positional, relu_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff // 2)
        self.w_1p = nn.Linear(d_positional, d_ff // 2)
        self.w_2c = nn.Linear(d_ff // 2, self.d_content)
        self.w_2p = nn.Linear(d_ff // 2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x
        xc = x[:, :self.d_content]
        xp = x[:, self.d_content:]

        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)

        output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)


#
class MultiLevelEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings_list,
                 d_embedding,
                 hparams,
                 d_positional=None,
                 max_len=300,
                 normalize=True,
                 dropout=0.1,
                 timing_dropout=0.0,
                 emb_dropouts_list=None,
                 extra_content_dropout=None,
                 word_table_np=None,
                 **kwargs):
        super().__init__()

        self.d_embedding = d_embedding
        self.partitioned = d_positional is not None
        self.hparams = hparams

        if self.partitioned:
            self.d_positional = d_positional
            self.d_content = self.d_embedding - self.d_positional
        else:
            self.d_positional = self.d_embedding
            self.d_content = self.d_embedding

        if emb_dropouts_list is None:
            emb_dropouts_list = [0.0] * len(num_embeddings_list)
        assert len(emb_dropouts_list) == len(num_embeddings_list)

        if word_table_np is not None:
            self.pretrain_dim = word_table_np.shape[1]
        else:
            self.pretrain_dim = 0

        embs = []
        emb_dropouts = []
        cun = len(num_embeddings_list) * 2
        for i, (num_embeddings, emb_dropout) in enumerate(zip(num_embeddings_list, emb_dropouts_list)):
            if hparams.use_cat:
                if i == len(num_embeddings_list) - 1:
                    # last is word
                    emb = nn.Embedding(num_embeddings, self.d_content // cun - self.pretrain_dim, **kwargs)
                else:
                    emb = nn.Embedding(num_embeddings, self.d_content // cun, **kwargs)
            else:
                emb = nn.Embedding(num_embeddings, self.d_content - self.pretrain_dim, **kwargs)
            embs.append(emb)
            emb_dropout = FeatureDropout(emb_dropout)
            emb_dropouts.append(emb_dropout)

        if word_table_np is not None:
            self.pretrain_emb = nn.Embedding(word_table_np.shape[0], self.pretrain_dim)
            self.pretrain_emb.weight.data.copy_(torch.from_numpy(word_table_np))
            self.pretrain_emb.weight.requires_grad_(False)
            self.pretrain_emb_dropout = FeatureDropout(0.33)

        self.embs = nn.ModuleList(embs)
        self.emb_dropouts = nn.ModuleList(emb_dropouts)

        if extra_content_dropout is not None:
            self.extra_content_dropout = FeatureDropout(extra_content_dropout)
        else:
            self.extra_content_dropout = None

        if normalize:
            self.layer_norm = LayerNormalization(d_embedding)
        else:
            self.layer_norm = lambda x: x

        self.dropout = FeatureDropout(dropout)
        self.timing_dropout = FeatureDropout(timing_dropout)

        # Learned embeddings
        self.max_len = max_len
        self.position_table = nn.Parameter(torch.FloatTensor(max_len, self.d_positional))
        init.normal_(self.position_table)

    def forward(self, xs, pre_words_idxs, batch_idxs, extra_content_annotations=None, batch=None, batched_inp=None,
                **kwargs):
        B, T, C = batched_inp.size()
        # extra_content_annotations = batched_inp
        content_annotations = [
            emb_dropout(emb(x), batch_idxs)
            for x, emb, emb_dropout in zip(xs, self.embs, self.emb_dropouts)
        ]
        if self.hparams.use_cat:
            content_annotations = torch.cat(content_annotations, dim=-1)
        else:
            content_annotations = sum(content_annotations)
        if self.pretrain_dim != 0:
            content_annotations = torch.cat(
                [content_annotations, self.pretrain_emb_dropout(self.pretrain_emb(pre_words_idxs))], dim=1)

        if extra_content_annotations is not None:
            if self.extra_content_dropout is not None:
                extra_content_annotations = self.extra_content_dropout(extra_content_annotations)

            if self.hparams.use_cat:
                content_annotations = torch.cat(
                    [content_annotations, extra_content_annotations], dim=-1)
            else:
                content_annotations += extra_content_annotations

        mask = batch['mask']
        timing_signal = self.position_table[:T, :].unsqueeze(0).expand_as(batched_inp)[mask]
        timing_signal = self.timing_dropout(timing_signal, batch_idxs)

        # Combine the content and timing signals
        if self.partitioned:
            annotations = torch.cat([content_annotations, timing_signal], 1)
        else:
            annotations = content_annotations + timing_signal

        # print(annotations.shape)
        annotations = self.layer_norm(self.dropout(annotations, batch_idxs))
        content_annotations = self.dropout(content_annotations, batch_idxs)

        return annotations, content_annotations, timing_signal, batch_idxs


#
class BiLinear(nn.Module):
    """Bi-linear layer"""

    def __init__(self, left_features, right_features, out_features, bias=True):
        '''

        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        '''
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = nn.Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = nn.Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = nn.Parameter(torch.Tensor(self.out_features, self.left_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        """

        Args:
          input_left: Tensor
        the left input tensor with shape = [batch1, batch2, ..., left_features]
          input_right: Tensor
        the right input tensor with shape = [batch1, batch2, ..., right_features]

        Returns:

        """
        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(-1, self.left_features)
        input_right = input_right.view(-1, self.right_features)

        # output [batch, out_features]
        output = nn.functional.bilinear(input_left, input_right, self.U, self.bias)
        output = output + nn.functional.linear(input_left, self.W_l, None) + nn.functional.linear(input_right, self.W_r,
                                                                                                  None)
        # convert back to [batch1, batch2, ..., out_features]
        return output


#
class BiAAttention(nn.Module):
    """Bi-Affine attention layer."""

    def __init__(self, hparams):
        super(BiAAttention, self).__init__()
        self.hparams = hparams

        self.dep_weight = nn.Parameter(torch.FloatTensor(hparams.d_biaffine + 1, hparams.d_biaffine + 1))
        nn.init.xavier_uniform_(self.dep_weight)

    def forward(self, input_d, input_e, input_s=None):
        device = input_d.device
        score = torch.matmul(torch.cat(
            [input_d, torch.FloatTensor(input_d.size(0), 1).to(device).fill_(1).requires_grad_(False)],
            dim=1), self.dep_weight)
        score1 = torch.matmul(score, torch.transpose(torch.cat(
            [input_e, torch.FloatTensor(input_e.size(0), 1).to(device).fill_(1).requires_grad_(False)],
            dim=1), 0, 1))

        return score1


class Dep_score(nn.Module):
    def __init__(self, hparams, num_labels):
        super(Dep_score, self).__init__()

        self.dropout_out = nn.Dropout2d(p=0.33)
        self.hparams = hparams
        out_dim = hparams.d_biaffine  # d_biaffine
        self.arc_h = nn.Linear(hparams.annotation_dim, hparams.d_biaffine)
        self.arc_c = nn.Linear(hparams.annotation_dim, hparams.d_biaffine)

        self.attention = BiAAttention(hparams)

        self.type_h = nn.Linear(hparams.annotation_dim, hparams.d_label_hidden)
        self.type_c = nn.Linear(hparams.annotation_dim, hparams.d_label_hidden)
        self.bilinear = BiLinear(hparams.d_label_hidden, hparams.d_label_hidden, num_labels)

    def forward(self, outputs, outpute):
        # output from rnn [batch, length, hidden_size]

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        outpute = self.dropout_out(outpute.transpose(1, 0)).transpose(1, 0)
        outputs = self.dropout_out(outputs.transpose(1, 0)).transpose(1, 0)

        # output size [batch, length, arc_space]
        arc_h = nn.functional.relu(self.arc_h(outputs))
        arc_c = nn.functional.relu(self.arc_c(outpute))

        # output size [batch, length, type_space]
        type_h = nn.functional.relu(self.type_h(outputs))
        type_c = nn.functional.relu(self.type_c(outpute))

        # apply dropout
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=0)
        type = torch.cat([type_h, type_c], dim=0)

        arc = self.dropout_out(arc.transpose(1, 0)).transpose(1, 0)
        arc_h, arc_c = arc.chunk(2, 0)

        type = self.dropout_out(type.transpose(1, 0)).transpose(1, 0)
        type_h, type_c = type.chunk(2, 0)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        out_arc = self.attention(arc_h, arc_c)
        out_type = self.bilinear(type_h, type_c)

        return out_arc, out_type


class LabelAttention(nn.Module):
    """Single-head Attention layer for label-specific representations"""

    def __init__(self, hparams, d_model, d_k, d_v, d_l, d_proj, use_resdrop=True, q_as_matrix=False,
                 residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
        super(LabelAttention, self).__init__()
        self.hparams = hparams
        self.d_k = d_k
        self.d_v = d_v
        self.d_l = d_l  # Number of Labels
        self.d_model = d_model  # Model Dimensionality
        self.d_proj = d_proj  # Projection dimension of each label output
        self.use_resdrop = use_resdrop  # Using Residual Dropout?
        self.q_as_matrix = q_as_matrix  # Using a Matrix of Q to be multiplied with input instead of learned q vectors
        self.combine_as_self = hparams.lal_combine_as_self  # Using the Combination Method of Self-Attention

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            if self.q_as_matrix:
                self.w_qs1 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
            else:
                self.w_qs1 = nn.Parameter(torch.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
            self.w_ks1 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_content, d_k // 2), requires_grad=True)
            self.w_vs1 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_content, d_v // 2), requires_grad=True)

            if self.q_as_matrix:
                self.w_qs2 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_positional, d_k // 2),
                                          requires_grad=True)
            else:
                self.w_qs2 = nn.Parameter(torch.FloatTensor(self.d_l, d_k // 2), requires_grad=True)
            self.w_ks2 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_positional, d_k // 2), requires_grad=True)
            self.w_vs2 = nn.Parameter(torch.FloatTensor(self.d_l, self.d_positional, d_v // 2), requires_grad=True)

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            if self.q_as_matrix:
                self.w_qs = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
            else:
                self.w_qs = nn.Parameter(torch.FloatTensor(self.d_l, d_k), requires_grad=True)
            self.w_ks = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_k), requires_grad=True)
            self.w_vs = nn.Parameter(torch.FloatTensor(self.d_l, d_model, d_v), requires_grad=True)

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        if self.combine_as_self:
            self.layer_norm = LayerNormalization(d_model)
        else:
            self.layer_norm = LayerNormalization(self.d_proj)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            if self.combine_as_self:
                self.proj = nn.Linear(self.d_l * d_v, d_model, bias=False)
            else:
                self.proj = nn.Linear(d_v, d_model, bias=False)  # input dimension does not match, should be d_l * d_v
        else:
            if self.combine_as_self:
                self.proj1 = nn.Linear(self.d_l * (d_v // 2), self.d_content, bias=False)
                self.proj2 = nn.Linear(self.d_l * (d_v // 2), self.d_positional, bias=False)
            else:
                self.proj1 = nn.Linear(d_v // 2, self.d_content, bias=False)
                self.proj2 = nn.Linear(d_v // 2, self.d_positional, bias=False)
        if not self.combine_as_self:
            self.reduce_proj = nn.Linear(d_model, self.d_proj, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, k_inp=None):
        len_inp = inp.size(0)
        v_inp_repeated = inp.repeat(self.d_l, 1).view(self.d_l, -1, inp.size(-1))  # d_l x len_inp x d_model
        if k_inp is None:
            k_inp_repeated = v_inp_repeated
        else:
            k_inp_repeated = k_inp.repeat(self.d_l, 1).view(self.d_l, -1, k_inp.size(-1))  # d_l x len_inp x d_model

        if not self.partitioned:
            if self.q_as_matrix:
                q_s = torch.bmm(k_inp_repeated, self.w_qs)  # d_l x len_inp x d_k
            else:
                q_s = self.w_qs.unsqueeze(1)  # d_l x 1 x d_k
            k_s = torch.bmm(k_inp_repeated, self.w_ks)  # d_l x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs)  # d_l x len_inp x d_v
        else:
            if self.q_as_matrix:
                q_s = torch.cat([
                    torch.bmm(k_inp_repeated[:, :, :self.d_content], self.w_qs1),
                    torch.bmm(k_inp_repeated[:, :, self.d_content:], self.w_qs2),
                ], -1)
            else:
                q_s = torch.cat([
                    self.w_qs1.unsqueeze(1),
                    self.w_qs2.unsqueeze(1),
                ], -1)
            k_s = torch.cat([
                torch.bmm(k_inp_repeated[:, :, :self.d_content], self.w_ks1),
                torch.bmm(k_inp_repeated[:, :, self.d_content:], self.w_ks2),
            ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:, :, :self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:, :, self.d_content:], self.w_vs2),
            ], -1)
        return q_s, k_s, v_s

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs, mask):
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * B) x T x d
        # (along with masks for the attention and output)
        n_head = self.d_l
        d_k, d_v = self.d_k, self.d_v

        T = batch_idxs.max_len
        B = batch_idxs.batch_size
        if self.q_as_matrix:
            q_padded = q_s.new_zeros((n_head, B, T, d_k))
            for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
                q_padded[:, i, :end - start, :] = q_s[:, start:end, :]
        else:
            q_padded = q_s.repeat(B, 1, 1)  # (d_l * B) x 1 x d_k
        k_padded = MultiHeadAttention.pad_seuence(k_s, batch_idxs)
        v_padded = MultiHeadAttention.pad_seuence(v_s, batch_idxs)

        if self.q_as_matrix:
            q_padded = q_padded.view(-1, T, d_k)
            attn_mask = ~mask.unsqueeze(1).expand(B, T, T).repeat(n_head, 1, 1)
        else:
            attn_mask = ~mask.unsqueeze(1).repeat(n_head, 1, 1)

        output_mask = mask.repeat(n_head, 1)

        return (
            q_padded,
            k_padded.view(-1, T, d_k),
            v_padded.view(-1, T, d_v),
            attn_mask,
            output_mask,
        )

    def combine_v(self, outputs):
        # Combine attention information from the different labels
        d_l = self.d_l
        outputs = outputs.view(d_l, -1, self.d_v)  # d_l x len_inp x d_v

        if not self.partitioned:
            # Switch from d_l x len_inp x d_v to len_inp x d_l x d_v
            if self.combine_as_self:
                outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, d_l * self.d_v)
            else:
                outputs = torch.transpose(outputs, 0, 1)  # .contiguous() #.view(-1, d_l * self.d_v)
            # Project back to residual size
            outputs = self.proj(outputs)  # Becomes len_inp x d_l x d_model
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:, :, :d_v1]
            outputs2 = outputs[:, :, d_v1:]
            if self.combine_as_self:
                outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, d_l * d_v1)
                outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, d_l * d_v1)
            else:
                outputs1 = torch.transpose(outputs1, 0, 1)  # .contiguous() #.view(-1, d_l * d_v1)
                outputs2 = torch.transpose(outputs2, 0, 1)  # .contiguous() #.view(-1, d_l * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
            ], -1)  # .contiguous()

        return outputs

    def forward(self, inp, batch_idxs, k_inp=None, batch=None, batched_inp=None, **kwargs):
        mask = batch['mask']
        residual = inp  # len_inp x d_model
        len_inp = inp.size(0)

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, k_inp=k_inp)
        # d_l x len_inp x d_k
        # q_s is d_l x 1 x d_k

        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs, mask)
        # q_padded, k_padded, v_padded: (d_l * batch_size) x max_len x d_kv
        # q_s is (d_l * batch_size) x 1 x d_kv

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask,
        )
        # outputs_padded: (d_l * batch_size) x max_len x d_kv
        # in LAL: (d_l * batch_size) x 1 x d_kv
        # on the best model, this is one value vector per label that is repeated max_len times
        if not self.q_as_matrix:
            outputs_padded = outputs_padded.repeat(1, output_mask.size(-1), 1)
        outputs = outputs_padded[output_mask]
        # outputs: (d_l * len_inp) x d_kv or LAL: (d_l * len_inp) x d_kv
        # output_mask: (d_l * batch_size) x max_len
        # torch.cuda.empty_cache()
        outputs = self.combine_v(outputs)
        # outputs: len_inp x d_l x d_model, whereas a normal self-attention layer gets len_inp x d_model
        if self.use_resdrop:
            if self.combine_as_self:
                outputs = self.residual_dropout(outputs, batch_idxs)
            else:
                outputs = torch.cat(
                    [self.residual_dropout(outputs[:, i, :], batch_idxs).unsqueeze(1) for i in range(self.d_l)], 1)
        if self.combine_as_self:
            outputs = self.layer_norm(outputs + inp)
        else:
            outputs = outputs + inp.unsqueeze(1)
            outputs = self.reduce_proj(outputs)  # len_inp x d_l x d_proj
            outputs = self.layer_norm(outputs)  # len_inp x d_l x d_proj
            outputs = outputs.view(len_inp, -1).contiguous()  # len_inp x (d_l * d_proj)

        return outputs, attns_padded


class Encoder(nn.Module):
    def __init__(self, hparams, embedding,
                 num_layers=1, num_heads=2, d_kv=32, d_ff=1024, d_l=112,
                 d_positional=None,
                 num_layers_position_only=0,
                 relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1,
                 use_lal=True,
                 lal_d_kv=128,
                 lal_d_proj=128,
                 lal_resdrop=True,
                 lal_pwff=True,
                 lal_q_as_matrix=False,
                 lal_partitioned=True):
        super().__init__()
        self.embedding_container = [embedding]
        d_model = embedding.d_embedding
        self.hparams = hparams

        d_k = d_v = d_kv

        self.stacks = []

        for i in range(hparams.num_layers):
            attn = MultiHeadAttention(hparams, num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout,
                                      attention_dropout=attention_dropout, d_positional=d_positional)
            if d_positional is None:
                ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout,
                                             residual_dropout=residual_dropout)
            else:
                ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout,
                                                        residual_dropout=residual_dropout)

            self.add_module(f"attn_{i}", attn)
            self.add_module(f"ff_{i}", ff)

            self.stacks.append((attn, ff))

        if use_lal:
            lal_d_positional = d_positional if lal_partitioned else None
            attn = LabelAttention(hparams, d_model, lal_d_kv, lal_d_kv, d_l, lal_d_proj, use_resdrop=lal_resdrop,
                                  q_as_matrix=lal_q_as_matrix,
                                  residual_dropout=residual_dropout, attention_dropout=attention_dropout,
                                  d_positional=lal_d_positional)
            ff_dim = lal_d_proj * d_l
            if hparams.lal_combine_as_self:
                ff_dim = d_model
            if lal_pwff:
                if d_positional is None or not lal_partitioned:
                    ff = PositionwiseFeedForward(ff_dim, d_ff, relu_dropout=relu_dropout,
                                                 residual_dropout=residual_dropout)
                else:
                    ff = PartitionedPositionwiseFeedForward(ff_dim, d_ff, d_positional, relu_dropout=relu_dropout,
                                                            residual_dropout=residual_dropout)
            else:
                ff = None

            self.add_module(f"attn_{num_layers}", attn)
            self.add_module(f"ff_{num_layers}", ff)
            self.stacks.append((attn, ff))

        self.num_layers_position_only = num_layers_position_only
        if self.num_layers_position_only > 0:
            assert d_positional is None, "num_layers_position_only and partitioned are incompatible"

    def forward(self, xs, pre_words_idxs, batch_idxs, extra_content_annotations=None, batch=None, **kwargs):
        emb = self.embedding_container[0]
        res, res_c, timing_signal, batch_idxs = emb(xs, pre_words_idxs, batch_idxs,
                                                    extra_content_annotations=extra_content_annotations,
                                                    batch=batch, **kwargs)

        for i, (attn, ff) in enumerate(self.stacks):
            res, current_attns = attn(res, batch_idxs, batch=batch, **kwargs)
            if ff is not None:
                res = ff(res, batch_idxs)

        return res, current_attns  # batch_idxs


class ChartParser(nn.Module):
    def __init__(
            self,
            embed: nn.Module,
            tag_vocab,
            label_vocab,
            type_vocab,
            config,
    ):
        super().__init__()
        self.embed = embed
        self.tag_vocab = tag_vocab
        self.label_vocab = label_vocab
        self.label_vocab_size = len(label_vocab)
        self.type_vocab = type_vocab

        self.hparams = config
        self.d_model = config.d_model
        self.partitioned = config.partitioned
        self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
        self.d_positional = (config.d_model // 2) if self.partitioned else None

        self.use_lal = config.use_lal
        if self.use_lal:
            self.lal_d_kv = config.lal_d_kv
            self.lal_d_proj = config.lal_d_proj
            self.lal_resdrop = config.lal_resdrop
            self.lal_pwff = config.lal_pwff
            self.lal_q_as_matrix = config.lal_q_as_matrix
            self.lal_partitioned = config.lal_partitioned
            self.lal_combine_as_self = config.lal_combine_as_self

        self.contributions = False

        num_embeddings_map = {
            'tags': len(tag_vocab),
        }
        emb_dropouts_map = {
            'tags': config.tag_emb_dropout,
        }

        self.emb_types = []
        if config.use_tags:
            self.emb_types.append('tags')
        if config.use_words:
            self.emb_types.append('words')

        self.use_tags = config.use_tags

        self.morpho_emb_dropout = None

        self.char_encoder = None
        self.elmo = None
        self.bert = None
        self.xlnet = None
        self.pad_left = config.pad_left
        self.roberta = None
        ex_dim = self.d_content
        if self.hparams.use_cat:
            cun = 0
            if config.use_words or config.use_tags:
                ex_dim = ex_dim // 2  # word dim = self.d_content/2
            if config.use_chars_lstm:
                cun = cun + 1
            if config.use_elmo or config.use_bert or config.use_xlnet:
                cun = cun + 1
            if cun > 0:
                ex_dim = ex_dim // cun

        self.project_xlnet = nn.Linear(embed.get_output_dim(), ex_dim, bias=False)

        if not config.dont_use_encoder:
            word_table_np = None

            self.embedding = MultiLevelEmbedding(
                [num_embeddings_map[emb_type] for emb_type in self.emb_types],
                config.d_model,
                hparams=config,
                d_positional=self.d_positional,
                dropout=config.embedding_dropout,
                timing_dropout=config.timing_dropout,
                emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
                extra_content_dropout=self.morpho_emb_dropout,
                max_len=config.sentence_max_len,
                word_table_np=word_table_np,
            )

            self.encoder = Encoder(
                config,
                self.embedding,
                d_l=len(label_vocab) - 1,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_kv=config.d_kv,
                d_ff=config.d_ff,
                d_positional=self.d_positional,
                relu_dropout=config.relu_dropout,
                residual_dropout=config.residual_dropout,
                attention_dropout=config.attention_dropout,
                use_lal=config.use_lal,
                lal_d_kv=config.lal_d_kv,
                lal_d_proj=config.lal_d_proj,
                lal_resdrop=config.lal_resdrop,
                lal_pwff=config.lal_pwff,
                lal_q_as_matrix=config.lal_q_as_matrix,
                lal_partitioned=config.lal_partitioned,
            )
        else:
            self.embedding = None
            self.encoder = None

        label_vocab_size = len(label_vocab)
        annotation_dim = ((label_vocab_size - 1) * self.lal_d_proj) if (
                self.use_lal and not self.lal_combine_as_self) else config.d_model
        # annotation_dim = self.encoder.stacks[-1][1].w_2c.out_features + self.encoder.stacks[-1][1].w_2p.out_features
        # annotation_dim = min((self.label_vocab_size - 1) * self.lal_d_proj, self.encoder.stacks[-1][1].w_2c.out_features + self.encoder.stacks[-1][1].w_2p.out_features)
        config.annotation_dim = annotation_dim

        self.f_label = nn.Sequential(
            nn.Linear(annotation_dim, config.d_label_hidden),
            LayerNormalization(config.d_label_hidden),
            nn.ReLU(),
            nn.Linear(config.d_label_hidden, label_vocab_size - 1),
        )
        self.dep_score = Dep_score(config, len(type_vocab))
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
        self.loss_funt = torch.nn.CrossEntropyLoss(reduction='sum')

        if not config.use_tags and hasattr(config, 'd_tag_hidden'):
            self.f_tag = nn.Sequential(
                nn.Linear(annotation_dim, config.d_tag_hidden),
                LayerNormalization(config.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(config.d_tag_hidden, tag_vocab.size),
            )
            self.tag_loss_scale = config.tag_loss_scale
        else:
            self.f_tag = None

    def forward(self, batch: dict):
        # sentences = batch['token']
        token_length: torch.LongTensor = batch['token_length']
        batch['mask'] = mask = util.lengths_to_mask(token_length)
        B, T = mask.size()
        golds: List[InternalParseNode] = batch.get('hpsg', None) if self.training else None
        if golds:
            sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in golds]
        else:
            sentences = [list(zip(t, w)) for w, t in zip(batch['FORM'], batch['CPOS'])]
        is_train = golds is not None

        packed_len = sum(token_length)
        i = 0
        batch_idxs = torch.arange(B).unsqueeze(1).expand_as(mask)[mask]
        batch_idxs = BatchIndices(batch_idxs)

        self.train(is_train)
        torch.set_grad_enabled(is_train)
        self.current_attns = None

        if golds is None:
            golds = [None] * len(sentences)

        extra_content_annotations_list = []

        features_packed = self.embed(batch)
        # For now, just project the features from the last word piece in each word
        extra_content_annotations = self.project_xlnet(features_packed)

        if self.encoder is not None:
            if len(extra_content_annotations_list) > 1:
                if self.hparams.use_cat:
                    extra_content_annotations = torch.cat(extra_content_annotations_list, dim=-1)
                else:
                    extra_content_annotations = sum(extra_content_annotations_list)
            elif len(extra_content_annotations_list) == 1:
                extra_content_annotations = extra_content_annotations_list[0]

            annotations, self.current_attns = self.encoder([batch['pos_id'][mask]], None, batch_idxs,
                                                           extra_content_annotations=extra_content_annotations[mask],
                                                           batch=batch,
                                                           batched_inp=extra_content_annotations)

            if self.partitioned and not self.use_lal:
                annotations = torch.cat([
                    annotations[:, 0::2],
                    annotations[:, 1::2],
                ], 1)

            if self.use_lal and not self.lal_combine_as_self:
                half_dim = self.lal_d_proj // 2
                annotations_3d = annotations.view(annotations.size(0), -1, half_dim)
                fencepost_annotations = torch.cat(
                    [annotations_3d[:-1, 0::2, :].flatten(1), annotations_3d[1:, 1::2, :].flatten(1)], dim=-1)
            else:
                fencepost_annotations = torch.cat([
                    annotations[:-1, :self.d_model // 2],
                    annotations[1:, self.d_model // 2:],
                ], 1)

            fencepost_annotations_start = fencepost_annotations
            fencepost_annotations_end = fencepost_annotations

        else:
            raise NotImplementedError()

        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        if not is_train:
            trees = []
            scores = []
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                tree, score = self.parse_from_annotations(fencepost_annotations_start[start:end, :],
                                                          fencepost_annotations_end[start:end, :], sentences[i], i)
                trees.append(tree)
                scores.append(score)

            return trees, scores

        pis = []
        pjs = []
        plabels = []
        paugment_total = 0.0
        cun = 0
        num_p = 0
        gis = []
        gjs = []
        glabels = []
        with torch.no_grad():
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                p_i, p_j, p_label, p_augment, g_i, g_j, g_label \
                    = self.parse_from_annotations(fencepost_annotations_start[start:end, :],
                                                  fencepost_annotations_end[start:end, :], sentences[i], i,
                                                  gold=golds[i])

                paugment_total += p_augment
                num_p += p_i.shape[0]
                pis.append(p_i + start)
                pjs.append(p_j + start)
                gis.append(g_i + start)
                gjs.append(g_j + start)
                plabels.append(p_label)
                glabels.append(g_label)

        device = annotations.device
        cells_i = torch.tensor(np.concatenate(pis + gis), device=device)
        cells_j = torch.tensor(np.concatenate(pjs + gjs), device=device)
        cells_label = torch.tensor(np.concatenate(plabels + glabels), device=device)

        cells_label_scores = self.f_label(fencepost_annotations_end[cells_j] - fencepost_annotations_start[cells_i])
        cells_label_scores = torch.cat([
            cells_label_scores.new_zeros((cells_label_scores.size(0), 1)),
            cells_label_scores
        ], 1)
        cells_label_scores = torch.gather(cells_label_scores, 1, cells_label[:, None])
        loss = cells_label_scores[:num_p].sum() - cells_label_scores[num_p:].sum() + paugment_total

        cun = 0
        for snum, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
            # [start,....,end-1]->[<root>,1, 2,...,n]
            leng = end - start
            arc_score, type_score = self.dep_score(fencepost_annotations_start[start:end, :],
                                                   fencepost_annotations_end[start:end, :])
            # arc_gather = gfather[cun] - start
            arc_gather = [leaf.father for leaf in golds[snum].leaves()]
            type_gather = [self.type_vocab.get_idx(leaf.type) for leaf in golds[snum].leaves()]
            cun += 1
            assert len(arc_gather) == leng - 1
            arc_score = torch.transpose(arc_score, 0, 1)
            loss = loss + 0.5 * self.loss_func(arc_score[1:, :], torch.tensor(arc_gather, device=device)) \
                   + 0.5 * self.loss_funt(type_score[1:, :], torch.tensor(type_gather, device=device))

        return None, loss

    def label_scores_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end):

        span_features = (torch.unsqueeze(fencepost_annotations_end, 0)
                         - torch.unsqueeze(fencepost_annotations_start, 1))

        if self.contributions and self.use_lal:
            contributions = np.zeros(
                (span_features.shape[0], span_features.shape[1], span_features.shape[2] // self.lal_d_proj))
            half_vector = span_features.shape[-1] // 2
            half_dim = self.lal_d_proj // 2
            for i in range(contributions.shape[0]):
                for j in range(contributions.shape[1]):
                    for l in range(contributions.shape[-1]):
                        contributions[i, j, l] = span_features[i, j,
                                                 l * half_dim:(l + 1) * half_dim].sum() + span_features[i, j,
                                                                                          half_vector + l * half_dim:half_vector + (
                                                                                                  l + 1) * half_dim].sum()
                    contributions[i, j, :] = (contributions[i, j, :] - np.min(contributions[i, j, :]))
                    contributions[i, j, :] = (contributions[i, j, :]) / (
                            np.max(contributions[i, j, :]) - np.min(contributions[i, j, :]))
                    # contributions[i,j,:] = contributions[i,j,:]/np.sum(contributions[i,j,:])
            contributions = torch.softmax(torch.Tensor(contributions), -1)

        label_scores_chart = self.f_label(span_features)
        label_scores_chart = torch.cat([
            label_scores_chart.new_zeros((label_scores_chart.size(0), label_scores_chart.size(1), 1)),
            label_scores_chart
        ], 2)
        if self.contributions and self.use_lal:
            return label_scores_chart, contributions
        return label_scores_chart

    def parse_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end, sentence, sentence_idx,
                               gold=None):
        is_train = gold is not None
        contributions = None
        if self.contributions and self.use_lal:
            label_scores_chart, contributions = self.label_scores_from_annotations(fencepost_annotations_start,
                                                                                   fencepost_annotations_end)
        else:
            label_scores_chart = self.label_scores_from_annotations(fencepost_annotations_start,
                                                                    fencepost_annotations_end)
        label_scores_chart_np = label_scores_chart.cpu().data.numpy()

        if is_train:
            decoder_args = dict(
                sentence_len=len(sentence),
                label_scores_chart=label_scores_chart_np,
                gold=gold,
                label_vocab=self.label_vocab,
                is_train=is_train)

            p_score, p_i, p_j, p_label, p_augment = const_decoder.decode(False, **decoder_args)
            g_score, g_i, g_j, g_label, g_augment = const_decoder.decode(True, **decoder_args)
            return p_i, p_j, p_label, p_augment, g_i, g_j, g_label
        else:
            arc_score, type_score = self.dep_score(fencepost_annotations_start, fencepost_annotations_end)

            arc_score_dc = torch.transpose(arc_score, 0, 1)
            arc_dc_np = arc_score_dc.cpu().data.numpy()

            type_np = type_score.cpu().data.numpy()
            type_np = type_np[1:, :]  # remove root
            type = type_np.argmax(axis=1)
            return self.decode_from_chart(sentence, label_scores_chart_np, arc_dc_np, type, sentence_idx=sentence_idx,
                                          contributions=contributions)

    def decode_from_chart_batch(self, sentences, charts_np, golds=None):
        trees = []
        scores = []
        if golds is None:
            golds = [None] * len(sentences)
        for sentence, chart_np, gold in zip(sentences, charts_np, golds):
            tree, score = self.decode_from_chart(sentence, chart_np, gold)
            trees.append(tree)
            scores.append(score)
        return trees, scores

    def decode_from_chart(self, sentence, label_scores_chart_np, arc_dc_np, type, sentence_idx=None, gold=None,
                          contributions=None):

        decoder_args = dict(
            sentence_len=len(sentence),
            label_scores_chart=label_scores_chart_np * self.hparams.const_lada,
            type_scores_chart=arc_dc_np * (1.0 - self.hparams.const_lada),
            gold=gold,
            label_vocab=self.label_vocab,
            type_vocab=self.type_vocab,
            is_train=False)

        force_gold = (gold is not None)

        # The optimized cython decoder implementation doesn't actually
        # generate trees, only scores and span indices. When converting to a
        # tree, we assume that the indices follow a preorder traversal.

        score, p_i, p_j, p_label, p_father, p_type, _ = hpsg_decoder.decode(force_gold, **decoder_args)
        if contributions is not None:
            d_l = (self.label_vocab_size - 2)
            mb_size = (self.current_attns.shape[0] // d_l)
            print('SENTENCE', sentence)

        idx = -1
        type_idx_to_token = self.type_vocab.idx_to_token
        label_idx_to_token = self.label_vocab.idx_to_token

        def get_label(index):
            label = label_idx_to_token[index]
            if not label:
                return ()
            return tuple(label.split('\t'))

        def make_tree():
            nonlocal idx
            idx += 1
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = get_label(label_idx)
            if contributions is not None:
                if label_idx > 0:
                    print(i, sentence[i], j, sentence[j - 1], label, label_idx, contributions[i, j, label_idx - 1])
                    print("CONTRIBUTIONS")
                    print(list(enumerate(contributions[i, j])))
                    print("ATTENTION DIST")
                    print(torch.softmax(self.current_attns[sentence_idx::mb_size, 0, i:j + 1], -1))
            if (i + 1) >= j:
                tag, word = sentence[i]
                if type is not None:
                    tree = trees.LeafParseNode(int(i), tag, word, p_father[i], type_idx_to_token[type[i]])
                else:
                    tree = trees.LeafParseNode(int(i), tag, word, p_father[i], type_idx_to_token[p_type[i]])
                if label:
                    assert label[0] != Sub_Head
                    tree = trees.InternalParseNode(label, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if label and label[0] != Sub_Head:
                    return [trees.InternalParseNode(label, children)]
                else:
                    return children

        tree_list = make_tree()
        assert len(tree_list) == 1
        tree = tree_list[0]
        return tree, score
