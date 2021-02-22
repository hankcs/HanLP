# A modified version of the implementation from the following paper:
# TENER: Adapting Transformer Encoder for Named Entity Recognition
# Hang Yan, Bocao Deng, Xiaonan Li, Xipeng Qiu

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hanlp.common.structure import ConfigTracker


class RelativeSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.

    Args:
        embedding_dim: embedding size of each position
        padding_idx:
    Returns:

    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size % 2 == 0
        weights = self.get_embedding(
            init_size + 1,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('weights', weights)

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".

        Args:
          num_embeddings:
          embedding_dim:
          padding_idx:  (Default value = None)

        Returns:

        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings // 2, num_embeddings // 2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings // 2 + 1
        return emb

    def forward(self, inputs: Tensor):
        """Input is expected to be of size [bsz x seqlen].

        Args:
          inputs: Tensor:

        Returns:

        """
        bsz, seq_len = inputs.size()
        max_pos = self.padding_idx + seq_len
        if max_pos > self.origin_shift:
            # recompute/expand embeddings if needed
            weights = self.get_embedding(
                max_pos * 2,
                self.embedding_dim,
                self.padding_idx,
            )
            weights = weights.to(self.weights.device)
            del self.weights
            self.origin_shift = weights.size(0) // 2
            self.register_buffer('weights', weights)

        positions = torch.arange(-seq_len, seq_len).to(inputs.device).long() + self.origin_shift  # 2*seq_len
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed


class RelativeMultiHeadAttn(nn.Module):
    def __init__(self, in_features, num_heads, dropout, r_w_bias=None, r_r_bias=None, init_seq_length=1024,
                 k_as_x=True):
        """
        Args:
            in_features:
            num_heads:
            dropout:
            r_w_bias: n_head x head_dim or None
            r_r_bias: n_head x head_dim or None
            init_seq_length:
            k_as_x:
        """
        super().__init__()
        self.k_as_x = k_as_x
        if k_as_x:
            self.qv_linear = nn.Linear(in_features, in_features * 2, bias=False)
        else:
            self.qkv_linear = nn.Linear(in_features, in_features * 3, bias=False)
        self.n_head = num_heads
        self.head_dim = in_features // num_heads
        self.dropout_layer = nn.Dropout(dropout)
        self.pos_embed = RelativeSinusoidalPositionalEmbedding(self.head_dim, 0, init_seq_length)
        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(num_heads, in_features // num_heads)))
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(torch.zeros(num_heads, in_features // num_heads)))
        else:
            self.r_r_bias = r_r_bias  # r_r_bias就是v
            self.r_w_bias = r_w_bias  # r_w_bias就是u

    def forward(self, x, mask):
        """

        Args:
          x: batch_size x max_len x d_model
          mask: batch_size x max_len

        Returns:

        """

        batch_size, max_len, d_model = x.size()
        pos_embed = self.pos_embed(mask)  # l x head_dim

        if self.k_as_x:
            qv = self.qv_linear(x)  # batch_size x max_len x d_model2
            q, v = torch.chunk(qv, chunks=2, dim=-1)
            k = x.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        else:
            qkv = self.qkv_linear(x)  # batch_size x max_len x d_model3
            q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
            k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b x n x l x d

        rw_head_q = q + self.r_r_bias[:, None]
        AC = torch.einsum('bnqd,bnkd->bnqk', [rw_head_q, k])  # b x n x l x d, n是head

        D_ = torch.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        B_ = torch.einsum('bnqd,ld->bnql', q, pos_embed)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        E_ = torch.einsum('bnqd,ld->bnql', k, pos_embed)  # bsz x head x max_len x 2max_len, key对relative的bias
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        if self.k_as_x:
            BD = self._shift(BD)
            attn = AC + BD
        else:
            BDE = self._shift(BD) + self._transpose_shift(E_)
            attn = AC + BDE

        attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, d_model)  # b x n x l x d

        return v

    def _shift(self, BD):
        """类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        转换为
        0   1  2
        -1  0  1
        -2 -1  0

        Args:
          BD: batch_size x n_head x max_len x 2max_len

        Returns:
          batch_size x n_head x max_len x max_len

        """
        bsz, n_head, max_len, _ = BD.size()
        zero_pad = BD.new_zeros(bsz, n_head, max_len, 1)
        BD = torch.cat([BD, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)  # bsz x n_head x (2max_len+1) x max_len
        BD = BD.narrow(dim=2, start=0, length=2 * max_len) \
            .view(bsz, n_head, max_len, -1)  # bsz x n_head x 2max_len x max_len
        BD = BD.narrow(dim=-1, start=max_len, length=max_len)
        return BD

    def _transpose_shift(self, E):
        """类似
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200

        转换为
          0  -10   -200
          1   00   -100
          2   10    000

        Args:
          E: batch_size x n_head x max_len x 2max_len

        Returns:
          batch_size x n_head x max_len x max_len

        """
        bsz, n_head, max_len, _ = E.size()
        zero_pad = E.new_zeros(bsz, n_head, max_len, 1)
        # bsz x n_head x -1 x (max_len+1)
        E = torch.cat([E, zero_pad], dim=-1).view(bsz, n_head, -1, max_len)
        indice = (torch.arange(max_len) * 2 + 1).to(E.device)
        E = E.index_select(index=indice, dim=-2).transpose(-1, -2)  # bsz x n_head x max_len x max_len

        return E


class RelativeTransformerLayer(nn.Module):
    def __init__(self,
                 in_features,
                 num_heads=4,
                 feedforward_dim=256,
                 dropout=0.2,
                 dropout_attn=None,
                 after_norm=True,
                 k_as_x=True,
                 init_seq_length=1024):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.after_norm = after_norm
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        self.self_attn = RelativeMultiHeadAttn(in_features,
                                               num_heads,
                                               dropout=dropout_attn,
                                               init_seq_length=init_seq_length,
                                               k_as_x=k_as_x)
        self.ffn = nn.Sequential(nn.Linear(in_features, feedforward_dim),
                                 nn.LeakyReLU(),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.Linear(feedforward_dim, in_features),
                                 nn.Dropout(dropout, inplace=True))

    def forward(self, x, mask):
        """

        Args:
          x: batch_size x max_len x hidden_size
          mask: batch_size x max_len, 为0的地方为pad

        Returns:
          batch_size x max_len x hidden_size

        """
        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x = self.self_attn(x, mask)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x


class RelativeTransformer(nn.Module):
    def __init__(self,
                 in_features,
                 num_layers,
                 feedforward_dim,
                 num_heads,
                 dropout,
                 dropout_attn=None,
                 after_norm=True,
                 init_seq_length=1024,
                 k_as_x=True):
        super().__init__()
        self.layers = nn.ModuleList([
            RelativeTransformerLayer(in_features, feedforward_dim, num_heads, dropout, dropout_attn, after_norm,
                                     init_seq_length=init_seq_length, k_as_x=k_as_x)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Tensor):
        """

        Args:
          x: batch_size x max_len
          mask: batch_size x max_len. 有value的地方为1
          x: Tensor: 
          mask: Tensor: 

        Returns:

        """

        for layer in self.layers:
            x = layer(x, mask)
        return x


class RelativeTransformerEncoder(RelativeTransformer, ConfigTracker):
    def __init__(self,
                 in_features,
                 num_layers=2,
                 num_heads=4,
                 feedforward_dim=256,
                 dropout=0.1,
                 dropout_attn=0.1,
                 after_norm=True,
                 k_as_x=True,
                 ):
        super().__init__(in_features, num_layers, num_heads, feedforward_dim, dropout, dropout_attn, after_norm)
        ConfigTracker.__init__(self, locals())

    def get_output_dim(self):
        return self.config['in_features']
