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

from typing import Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from hanlp.layers.crf.crf import CRF


class RNNTaggingModel(nn.Module):

    def __init__(self,
                 embed: Union[nn.Embedding, int],
                 rnn_input,
                 rnn_hidden,
                 n_out,
                 drop=0.5,
                 crf=True,
                 crf_constraints=None):
        super(RNNTaggingModel, self).__init__()

        # the embedding layer
        if isinstance(embed, nn.Module):
            self.embed = embed
            n_embed = embed.embedding_dim
        else:
            self.embed = None
            n_embed = embed

        if rnn_input:
            self.embed_to_rnn = nn.Linear(n_embed, rnn_input)
        else:
            self.embed_to_rnn = None
            rnn_input = n_embed

        # the word-lstm layer
        self.word_lstm = nn.LSTM(input_size=rnn_input,
                                 hidden_size=rnn_hidden,
                                 batch_first=True,
                                 bidirectional=True)

        # the output layer
        self.out = nn.Linear(rnn_hidden * 2, n_out)
        # the CRF layer
        self.crf = CRF(n_out, crf_constraints) if crf else None

        self.drop = nn.Dropout(drop)
        # self.drop = SharedDropout(drop)
        # self.drop = LockedDropout(drop)

        self.reset_parameters()

    def reset_parameters(self):
        # init Linear
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self,
                x: torch.Tensor,
                batch=None,
                **kwargs):
        # get the mask and lengths of given batch
        mask = x.gt(0)
        lens = mask.sum(dim=1)
        # get outputs from embedding layers
        if isinstance(self.embed, nn.Embedding):
            x = self.embed(x[mask])
        else:
            x = self.embed(batch, mask=mask)
            if x.dim() == 3:
                x = x[mask]
        x = self.drop(x)
        if self.embed_to_rnn:
            x = self.embed_to_rnn(x)
        x = pack_sequence(torch.split(x, lens.tolist()), True)
        x, _ = self.word_lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.drop(x)

        return self.out(x), mask
