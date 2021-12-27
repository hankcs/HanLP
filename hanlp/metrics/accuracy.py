# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-12 17:56
from typing import Optional, Iterable

import torch

from hanlp.metrics.metric import Metric


class CategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    Copied from AllenNLP and added several methods.
    """

    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise ValueError(
                "Tie break in Categorical Accuracy can be done only for maximum (top_k = 1)"
            )
        if top_k <= 0:
            raise ValueError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(
            self,
            predictions: torch.Tensor,
            gold_labels: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ValueError(
                "gold_labels must have dimension == predictions.size() - 1 but "
                "found tensor of shape: {}".format(predictions.size())
            )
        if (gold_labels >= num_classes).any():
            raise ValueError(
                "A gold label passed to Categorical Accuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1)
            else:
                top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[
                torch.arange(gold_labels.numel(), device=gold_labels.device).long(), gold_labels
            ].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1)
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    @property
    def score(self):
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        return accuracy

    def __repr__(self) -> str:
        return f'Accuracy:{self.score:.2%}'

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0


class BooleanAccuracy(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.
    That is, if your prediction has shape (batch_size, dim_1, ..., dim_n), this metric considers that
    as a set of `batch_size` predictions and checks that each is *entirely* correct across the remaining dims.
    This means the denominator in the accuracy computation is `batch_size`, with the caveat that predictions
    that are totally masked are ignored (in which case the denominator is the number of predictions that have
    at least one unmasked element).

    This is similar to [`CategoricalAccuracy`](./categorical_accuracy.md), if you've already done a `.max()`
    on your predictions.  If you have categorical output, though, you should typically just use
    `CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    `CategoricalAccuracy`, which assumes a final dimension of size `num_classes`.
    """

    def __init__(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0.0

    def __call__(
            self,
            predictions: torch.Tensor,
            gold_labels: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predictions`.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predictions`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        if gold_labels.size() != predictions.size():
            raise ValueError(
                f"gold_labels must have shape == predictions.size() but "
                f"found tensor of shape: {gold_labels.size()}"
            )
        if mask is not None and mask.size() != predictions.size():
            raise ValueError(
                f"mask must have shape == predictions.size() but "
                f"found tensor of shape: {mask.size()}"
            )

        batch_size = predictions.size(0)

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

            # We want to skip predictions that are completely masked;
            # so we'll keep predictions that aren't.
            keep = mask.view(batch_size, -1).max(dim=1)[0]
        else:
            keep = torch.ones(batch_size, device=predictions.device).bool()

        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)

        # At this point, predictions is (batch_size, rest_of_dims_combined),
        # so .eq -> .prod will be 1 if every element of the instance prediction is correct
        # and 0 if at least one element of the instance prediction is wrong.
        # Because of how we're handling masking, masked positions are automatically "correct".
        correct = predictions.eq(gold_labels).prod(dim=1).float()

        # Since masked positions are correct, we need to explicitly exclude instance predictions
        # where the entire prediction is masked (because they look "correct").
        self._correct_count += (correct * keep).sum()
        self._total_count += keep.sum()

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated accuracy.
        """
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)
