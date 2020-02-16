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


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y`,
    in which :math:`x` and :math:`y` can be concatenated with bias terms.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


class Triaffine(nn.Module):
    r"""
    Triaffine layer for second-order scoring.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y`.
    Usually, :math:`x` and :math:`y` can be concatenated with bias terms.

    References:
        - Yu Zhang, Zhenghua Li and Min Zhang. 2020.
          `Efficient Second-Order TreeCRF for Neural Dependency Parsing`_.
        - Xinyu Wang, Jingxian Huang, and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.

    Args:
        n_in (int):
            The size of the input feature.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.

    .. _Efficient Second-Order TreeCRF for Neural Dependency Parsing:
        https://www.aclweb.org/anthology/2020.acl-main.302/
    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    """

    def __init__(self, n_in, bias_x=False, bias_y=False):
        super().__init__()

        self.n_in = n_in
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_in+bias_x, n_in, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y, z):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, seq_len, seq_len, seq_len]``.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        w = torch.einsum('bzk,ikj->bzij', z, self.weight)
        # [batch_size, seq_len, seq_len, seq_len]
        s = torch.einsum('bxi,bzij,byj->bzxy', x, w, y)

        return s
