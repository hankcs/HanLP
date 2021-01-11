# This file is modified from udify, which is licensed under the MIT license:
# MIT License
#
# Copyright (c) 2019 Dan Kondratyuk
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
"""
The dot-product "Layer Attention" that is applied to the layers of BERT, along with layer dropout to reduce overfitting
"""

from typing import List, Tuple

import torch
from torch.nn import ParameterList, Parameter

from hanlp.common.structure import ConfigTracker


class ScalarMixWithDropout(torch.nn.Module):
    """Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.
    
    If ``do_layer_norm=True`` then apply layer normalization to each tensor before weighting.
    
    If ``dropout > 0``, then for each scalar weight, adjust its softmax weight mass to 0 with
    the dropout probability (i.e., setting the unnormalized weight to -inf). This effectively
    should redistribute dropped probability mass to all other weights.

    Args:

    Returns:

    """

    def __init__(self,
                 mixture_range: Tuple[int, int],
                 do_layer_norm: bool = False,
                 initial_scalar_parameters: List[float] = None,
                 trainable: bool = True,
                 dropout: float = None,
                 dropout_value: float = -1e20,
                 **kwargs) -> None:
        super(ScalarMixWithDropout, self).__init__()
        self.mixture_range = mixture_range
        mixture_size = mixture_range[1] - mixture_range[0]
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        self.dropout = dropout

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ValueError("Length of initial_scalar_parameters {} differs "
                             "from mixture_size {}".format(
                initial_scalar_parameters, mixture_size))

        # self.scalar_parameters = ParameterList(
        #     [Parameter(torch.FloatTensor([initial_scalar_parameters[i]]),
        #                requires_grad=trainable) for i
        #      in range(mixture_size)])
        self.scalar_parameters = Parameter(torch.FloatTensor(initial_scalar_parameters), requires_grad=True)
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

        if self.dropout:
            dropout_mask = torch.zeros(len(self.scalar_parameters))
            dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(dropout_value)
            self.register_buffer("dropout_mask", dropout_mask)
            self.register_buffer("dropout_fill", dropout_fill)

    def forward(self, tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        
        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.
        
        When ``do_layer_norm=False`` the ``mask`` is ignored.

        Args:
          tensors: List[torch.Tensor]: 
          # pylint: disable:  (Default value = arguments-differmask: torch.Tensor = None)

        Returns:

        """
        if len(tensors) != self.mixture_size:
            tensors = tensors[self.mixture_range[0]:self.mixture_range[1]]
        if len(tensors) != self.mixture_size:
            raise ValueError("{} tensors were passed, but the module was initialized to "
                             "mix {} tensors.".format(len(tensors), self.mixture_size))

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2) / num_elements_not_masked
            return (tensor - mean) / torch.sqrt(variance + 1E-12)

        weights = self.scalar_parameters

        if self.dropout:
            weights = torch.where(self.dropout_mask.uniform_() > self.dropout, weights, self.dropout_fill)

        normed_weights = torch.nn.functional.softmax(weights, dim=0)

        if not self.do_layer_norm:
            return self.gamma * torch.einsum('i,ijkl->jkl', normed_weights, tensors)
            # pieces = []
            # for weight, tensor in zip(normed_weights, tensors):
            #     pieces.append(weight * tensor)
            # return self.gamma * sum(pieces)
        else:
            normed_weights = torch.split(normed_weights, split_size_or_sections=1)
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                                                      broadcast_mask, num_elements_not_masked))
            return self.gamma * sum(pieces)


class ScalarMixWithDropoutBuilder(ConfigTracker, ScalarMixWithDropout):

    def __init__(self,
                 mixture_range: Tuple[int, int],
                 do_layer_norm: bool = False,
                 initial_scalar_parameters: List[float] = None,
                 trainable: bool = True,
                 dropout: float = None,
                 dropout_value: float = -1e20) -> None:
        super().__init__(locals())

    def build(self):
        return ScalarMixWithDropout(**self.config)
