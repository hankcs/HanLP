# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-23 16:12
import torch

from hanlp.metrics.metric import Metric


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    argsort = x.argsort()
    ranks = torch.zeros_like(argsort, device=x.device)
    ranks[argsort] = torch.arange(len(x), device=x.device)
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors. Adopted from
    https://discuss.pytorch.org/t/spearmans-correlation/91931/5

    Args:
        x: Shape (N, )
        y: Shape (N, )

    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


class SpearmanCorrelation(Metric):
    """
    This `Metric` calculates the sample Spearman correlation coefficient (r)
    between two tensors. Each element in the two tensors is assumed to be
    a different observation of the variable (i.e., the input tensors are
    implicitly flattened into vectors and the correlation is calculated
    between the vectors).

    <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>
    """

    @property
    def score(self):
        return spearman_correlation(self.total_predictions, self.total_gold_labels).item()

    def __init__(self) -> None:
        super().__init__()
        self.total_predictions = torch.zeros(0)
        self.total_gold_labels = torch.zeros(0)

    def __call__(
            self,
            predictions: torch.Tensor,
            gold_labels: torch.Tensor,
            mask=None
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predictions`.
        """
        if mask is not None:
            raise NotImplemented('mask not supported in SpearmanCorrelation for now.')
        # Flatten predictions, gold_labels, and mask. We calculate the Spearman correlation between
        # the vectors, since each element in the predictions and gold_labels tensor is assumed
        # to be a separate observation.
        predictions = predictions.reshape(-1)
        gold_labels = gold_labels.reshape(-1)

        self.total_predictions = self.total_predictions.to(predictions.device)
        self.total_gold_labels = self.total_gold_labels.to(gold_labels.device)
        self.total_predictions = torch.cat((self.total_predictions, predictions), 0)
        self.total_gold_labels = torch.cat((self.total_gold_labels, gold_labels), 0)

    def reset(self):
        self.total_predictions = torch.zeros(0)
        self.total_gold_labels = torch.zeros(0)

    def __str__(self) -> str:
        return f'spearman: {self.score * 100:.2f}'
