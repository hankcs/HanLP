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
from torch import nn
from hanlp.components.parsers.constituency.treecrf import CRFConstituency
from hanlp.components.parsers.alg import cky
from hanlp.components.parsers.biaffine.biaffine import Biaffine
from hanlp.components.parsers.biaffine.mlp import MLP


class CRFConstituencyDecoder(nn.Module):
    r"""
    The implementation of CRF Constituency Parser,
    also called FANCY (abbr. of Fast and Accurate Neural Crf constituencY) Parser.

    References:
        - Yu Zhang, Houquan Zhou and Zhenghua Li. 2020.
          `Fast and Accurate Neural CRF Constituency Parsing`_.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_feats (int):
            The size of the feat vocabulary.
        n_labels (int):
            The number of labels.
        feat (str):
            Specifies which type of additional feature to use: ``'char'`` | ``'bert'`` | ``'tag'``.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'bert'``: BERT representations, other pretrained langugae models like XLNet are also feasible.
            ``'tag'``: POS tag embeddings.
            Default: 'char'.
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if ``feat='char'``. Default: 50.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'`` and ``'xlnet-base-cased'``.
            This is required if ``feat='bert'``. The full list can be found in `transformers`.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use. Required if ``feat='bert'``.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers. Required if ``feat='bert'``. Default: .0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .33.
        n_hidden (int):
            The size of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            The number of LSTM layers. Default: 3.
        lstm_dropout (float):
            The dropout ratio of LSTM. Default: .33.
        n_mlp_span (int):
            Span MLP size. Default: 500.
        n_mlp_label  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            The dropout ratio of MLP layers. Default: .33.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _Fast and Accurate Neural CRF Constituency Parsing:
        https://www.ijcai.org/Proceedings/2020/560/
    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_labels,
                 n_hidden=400,
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 **kwargs
                 ):
        super().__init__()

        # the MLP layers
        self.mlp_span_l = MLP(n_in=n_hidden, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_span_r = MLP(n_in=n_hidden, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_label_l = MLP(n_in=n_hidden, n_out=n_mlp_label, dropout=mlp_dropout)
        self.mlp_label_r = MLP(n_in=n_hidden, n_out=n_mlp_label, dropout=mlp_dropout)

        # the Biaffine layers
        self.span_attn = Biaffine(n_in=n_mlp_span, bias_x=True, bias_y=False)
        self.label_attn = Biaffine(n_in=n_mlp_label, n_out=n_labels, bias_x=True, bias_y=True)
        self.crf = CRFConstituency()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        r"""
        Args:
            x (~torch.FloatTensor): ``[batch_size, seq_len, hidden_dim]``.
                Hidden states from encoder.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible spans.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each span.
        """

        x_f, x_b = x.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        # apply MLPs to the BiLSTM output states
        span_l = self.mlp_span_l(x)
        span_r = self.mlp_span_r(x)
        label_l = self.mlp_label_l(x)
        label_r = self.mlp_label_r(x)

        # [batch_size, seq_len, seq_len]
        s_span = self.span_attn(span_l, span_r)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return s_span, s_label

    def loss(self, s_span, s_label, charts, mask, mbr=True):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all spans
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each span.
            charts (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels, in which positions without labels are filled with -1.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.
            mbr (bool):
                If ``True``, returns marginals for MBR decoding. Default: ``True``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The training loss and
                original span scores of shape ``[batch_size, seq_len, seq_len]`` if ``mbr=False``, or marginals otherwise.
        """

        span_mask = charts.ge(0) & mask
        span_loss, span_probs = self.crf(s_span, mask, span_mask, mbr)
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = span_loss + label_loss

        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        r"""
        Args:
            s_span (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all spans.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all labels on each span.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask for covering the unpadded tokens in each chart.

        Returns:
            list[list[tuple]]:
                Sequences of factorized labeled trees traversed in pre-order.
        """

        span_preds = cky(s_span, mask)
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j in spans] for spans, labels in zip(span_preds, label_preds)]


class CRFConstituencyModel(nn.Module):

    def __init__(self, encoder, decoder: CRFConstituencyDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        r"""
        Args:
            batch (~dict):
                Batch of input data.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len]`` holds scores of all possible spans.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each span.
        """
        x = self.encoder(batch)
        return self.decoder(x)
