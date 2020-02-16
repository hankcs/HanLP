# Adopted from https://github.com/airaria/TextBrewer
# Apache License Version 2.0
from abc import ABC, abstractmethod

import torch

# x is between 0 and 1
from hanlp_common.configurable import AutoConfigurable


def linear_growth_weight_scheduler(x):
    return x


def linear_decay_weight_scheduler(x):
    return 1 - x


def constant_temperature_scheduler(logits_S, logits_T, base_temperature):
    '''
    Remember to detach logits_S 
    '''
    return base_temperature


def flsw_temperature_scheduler_builder(beta, gamma, eps=1e-4, *args):
    '''
    adapted from arXiv:1911.07471
    '''

    def flsw_temperature_scheduler(logits_S, logits_T, base_temperature):
        v = logits_S.detach()
        t = logits_T.detach()
        with torch.no_grad():
            v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
            t = t / (torch.norm(t, dim=-1, keepdim=True) + eps)
            w = torch.pow((1 - (v * t).sum(dim=-1)), gamma)
            tau = base_temperature + (w.mean() - w) * beta
        return tau

    return flsw_temperature_scheduler


def cwsm_temperature_scheduler_builder(beta, *args):
    '''
    adapted from arXiv:1911.07471
    '''

    def cwsm_temperature_scheduler(logits_S, logits_T, base_temperature):
        v = logits_S.detach()
        with torch.no_grad():
            v = torch.softmax(v, dim=-1)
            v_max = v.max(dim=-1)[0]
            w = 1 / (v_max + 1e-3)
            tau = base_temperature + (w.mean() - w) * beta
        return tau

    return cwsm_temperature_scheduler


class LinearTeacherAnnealingScheduler(object):
    def __init__(self, num_training_steps: int) -> None:
        super().__init__()
        self._num_training_steps = num_training_steps
        self._current_training_steps = 0

    def step(self):
        self._current_training_steps += 1

    def __float__(self):
        return self._current_training_steps / self._num_training_steps


class TemperatureScheduler(ABC, AutoConfigurable):

    def __init__(self, base_temperature) -> None:
        super().__init__()
        self.base_temperature = base_temperature

    def __call__(self, logits_S, logits_T):
        return self.forward(logits_S, logits_T)

    @abstractmethod
    def forward(self, logits_S, logits_T):
        raise NotImplementedError()

    @staticmethod
    def from_name(name):
        classes = {
            'constant': ConstantScheduler,
            'flsw': FlswScheduler,
            'cwsm': CwsmScheduler,
        }
        assert name in classes, f'Unsupported temperature scheduler {name}. Expect one from {list(classes.keys())}.'
        return classes[name]()


class FunctionalScheduler(TemperatureScheduler):

    def __init__(self, scheduler_func, base_temperature) -> None:
        super().__init__(base_temperature)
        self._scheduler_func = scheduler_func

    def forward(self, logits_S, logits_T):
        return self._scheduler_func(logits_S, logits_T, self.base_temperature)


class ConstantScheduler(TemperatureScheduler):
    def forward(self, logits_S, logits_T):
        return self.base_temperature


class FlswScheduler(FunctionalScheduler):
    def __init__(self, beta=1, gamma=1, eps=1e-4, base_temperature=8):
        super().__init__(flsw_temperature_scheduler_builder(beta, gamma, eps), base_temperature)
        self.beta = beta
        self.gamma = gamma
        self.eps = eps


class CwsmScheduler(FunctionalScheduler):
    def __init__(self, beta=1, base_temperature=8):
        super().__init__(cwsm_temperature_scheduler_builder(beta), base_temperature)
        self.beta = beta
