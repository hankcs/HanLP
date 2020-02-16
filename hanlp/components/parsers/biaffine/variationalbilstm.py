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
import torch.nn as nn
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from hanlp.common.structure import ConfigTracker
from hanlp.layers.dropout import SharedDropout


class VariationalLSTM(nn.Module):
    r"""
    LSTM is an variant of the vanilla bidirectional LSTM adopted by Biaffine Parser
    with the only difference of the dropout strategy.
    It drops nodes in the LSTM layers (input and recurrent connections)
    and applies the same dropout mask at every recurrent timesteps.

    APIs are roughly the same as :class:`~torch.nn.LSTM` except that we only allows
    :class:`~torch.nn.utils.rnn.PackedSequence` as input.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    Args:
        input_size (int):
            The number of expected features in the input.
        hidden_size (int):
            The number of features in the hidden state `h`.
        num_layers (int):
            The number of recurrent layers. Default: 1.
        bidirectional (bool):
            If ``True``, becomes a bidirectional LSTM. Default: ``False``
        dropout (float):
            If non-zero, introduces a :class:`SharedDropout` layer on the outputs of each LSTM layer except the last layer.
            Default: 0.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_directions = 1 + self.bidirectional

        self.f_cells = nn.ModuleList()
        if bidirectional:
            self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            input_size = hidden_size * self.num_directions

        self.reset_parameters()

    def __repr__(self):
        s = f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.bidirectional:
            s += f", bidirectional={self.bidirectional}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size])) for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(self, sequence, hx=None):
        r"""
        Args:
            sequence (~torch.nn.utils.rnn.PackedSequence):
                A packed variable length sequence.
            hx (~torch.Tensor, ~torch.Tensor):
                A tuple composed of two tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial hidden state
                for each element in the batch.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial cell state
                for each element in the batch.
                If `hx` is not provided, both `h` and `c` default to zero.
                Default: ``None``.

        Returns:
            ~torch.nn.utils.rnn.PackedSequence, (~torch.Tensor, ~torch.Tensor):
                The first is a packed variable length sequence.
                The second is a tuple of tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the hidden state for `t=seq_len`.
                Like output, the layers can be separated using ``h.view(num_layers, num_directions, batch_size, hidden_size)``
                and similarly for c.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the cell state for `t=seq_len`.
        """
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = x.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        c = c.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            x_i, (h_i, c_i) = self.layer_forward(x=x,
                                                 hx=(h[i, 0], c[i, 0]),
                                                 cell=self.f_cells[i],
                                                 batch_sizes=batch_sizes)
            if self.bidirectional:
                x_b, (h_b, c_b) = self.layer_forward(x=x,
                                                     hx=(h[i, 1], c[i, 1]),
                                                     cell=self.b_cells[i],
                                                     batch_sizes=batch_sizes,
                                                     reverse=True)
                x_i = torch.cat((x_i, x_b), -1)
                h_i = torch.stack((h_i, h_b))
                c_i = torch.stack((c_i, c_b))
            x = x_i
            h_n.append(h_i)
            c_n.append(h_i)

        x = PackedSequence(x,
                           sequence.batch_sizes,
                           sequence.sorted_indices,
                           sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)

        return x, hx


class VariationalLSTMEncoder(VariationalLSTM, ConfigTracker):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bidirectional=False,
                 variational_dropout=0,
                 word_dropout=0,
                 ):
        super().__init__(input_size, hidden_size, num_layers, bidirectional, variational_dropout)
        ConfigTracker.__init__(self, locals())
        self.lstm_dropout = SharedDropout(p=word_dropout)

    # noinspection PyMethodOverriding
    def forward(self, embed, mask):
        batch_size, seq_len = mask.shape
        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = super().forward(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)
        return x

    def get_output_dim(self):
        return self.hidden_size * self.num_directions
