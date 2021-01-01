# This file is modified from udify and allennlp, which are licensed under the MIT license:
# MIT License
#
# Copyright (c) 2019 Dan Kondratyuk and allennlp
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

import os
from typing import List, Dict, Tuple, Union

import numpy
import torch


def get_ud_treebank_files(dataset_dir: str, treebanks: List[str] = None) -> Dict[str, Tuple[str, str, str]]:
    """Retrieves all treebank data paths in the given directory.
    Adopted from https://github.com/Hyperparticle/udify
    MIT Licence

    Args:
      dataset_dir: 
      treebanks: 
      dataset_dir: str: 
      treebanks: List[str]:  (Default value = None)

    Returns:

    
    """
    datasets = {}
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]

        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        test_file = [file for file in conllu_files if file.endswith("test.conllu")]

        train_file = os.path.join(treebank_path, train_file[0]) if train_file else None
        dev_file = os.path.join(treebank_path, dev_file[0]) if dev_file else None
        test_file = os.path.join(treebank_path, test_file[0]) if test_file else None

        datasets[treebank] = (train_file, dev_file, test_file)
    return datasets


def sequence_cross_entropy(log_probs: torch.FloatTensor,
                           targets: torch.LongTensor,
                           weights: torch.FloatTensor,
                           average: str = "batch",
                           label_smoothing: float = None) -> torch.FloatTensor:
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = log_probs.view(-1, log_probs.size(2))
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = log_probs.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss


def sequence_cross_entropy_with_logits(
        logits: torch.FloatTensor,
        targets: torch.LongTensor,
        weights: Union[torch.FloatTensor, torch.BoolTensor],
        average: str = "batch",
        label_smoothing: float = None,
        gamma: float = None,
        alpha: Union[float, List[float], torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the `torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.
    
    # Parameters
    
    logits : `torch.FloatTensor`, required.
        A `torch.FloatTensor` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : `torch.LongTensor`, required.
        A `torch.LongTensor` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : `Union[torch.FloatTensor, torch.BoolTensor]`, required.
        A `torch.FloatTensor` of size (batch, sequence_length)
    average: `str`, optional (default = `"batch"`)
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If `None`, return a vector
        of losses per batch element.
    label_smoothing : `float`, optional (default = `None`)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like `[0.05, 0.05, 0.85, 0.05]` if the 3rd class was
        the correct label.
    gamma : `float`, optional (default = `None`)
        Focal loss[*] focusing parameter `gamma` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        `gamma` is, the more focus on hard examples.
    alpha : `Union[float, List[float]]`, optional (default = `None`)
        Focal loss[*] weighting factor `alpha` to balance between classes. Can be
        used independently with `gamma`. If a single `float` is provided, it
        is assumed binary case using `alpha` and `1 - alpha` for positive and
        negative respectively. If a list of `float` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.
    
    # Returns
    
    `torch.FloatTensor`
        A torch.FloatTensor representing the cross entropy loss.
        If `average=="batch"` or `average=="token"`, the returned loss is a scalar.
        If `average is None`, the returned loss is a vector of shape (batch_size,).

    Args:
      logits: torch.FloatTensor: 
      targets: torch.LongTensor: 
      weights: Union[torch.FloatTensor: 
      torch.BoolTensor]: 
      average: str:  (Default value = "batch")
      label_smoothing: float:  (Default value = None)
      gamma: float:  (Default value = None)
      alpha: Union[float: 
      List[float]: 
      torch.FloatTensor]:  (Default value = None)

    Returns:

    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.to(logits.dtype)
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1.0 - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):

            # shape : (2,)
            alpha_factor = torch.tensor(
                [1.0 - float(alpha), float(alpha)], dtype=weights.dtype, device=weights.device
            )

        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):

            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)

            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(
                ("alpha must be float, list of float, or torch.FloatTensor, {} provided.").format(
                    type(alpha)
                )
            )
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(
            *targets.size()
        )
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(
            -1, targets_flat, 1.0 - label_smoothing
        )
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
                weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        num_non_empty_sequences = (weights_batch_sum > 0).sum() + tiny_value_of_dtype(
            negative_log_likelihood.dtype
        )
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (
                weights_batch_sum.sum() + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
                weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        return per_batch_loss


def tiny_value_of_dtype(dtype: torch.dtype):
    """Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.

    Args:
      dtype: torch.dtype: 

    Returns:

    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def combine_initial_dims_to_1d_or_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Given a (possibly higher order) tensor of ids with shape
    (d1, ..., dn, sequence_length)

    Args:
      tensor: torch.Tensor: 

    Returns:
      If original tensor is 1-d or 2-d, return it as is.

    """
    if tensor.dim() <= 2:
        return tensor
    else:
        return tensor.view(-1, tensor.size(-1))


def uncombine_initial_dims(tensor: torch.Tensor, original_size: torch.Size) -> torch.Tensor:
    """Given a tensor of embeddings with shape
    (d1 * ... * dn, sequence_length, embedding_dim)
    and the original shape
    (d1, ..., dn, sequence_length),

    Args:
      tensor: torch.Tensor: 
      original_size: torch.Size: 

    Returns:
      (d1, ..., dn, sequence_length, embedding_dim).
      If original size is 1-d or 2-d, return it as is.

    """
    if len(original_size) <= 2:
        return tensor
    else:
        view_args = list(original_size) + [tensor.size(-1)]
        return tensor.view(*view_args)


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.

    Args:
      size: int: 
      device: int: 

    Returns:

    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    """Returns the device of the tensor.

    Args:
      tensor: torch.Tensor: 

    Returns:

    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()
