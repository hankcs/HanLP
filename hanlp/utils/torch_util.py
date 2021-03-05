# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 15:52
import os
import random
import time
from typing import List, Union, Dict

import numpy as np
import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit, nvmlShutdown, nvmlDeviceGetCount
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from hanlp.utils.log_util import logger


def gpus_available() -> Dict[int, float]:
    try:
        nvmlInit()
        gpus = {}
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if visible_devices is None:
            visible_devices = list(range(nvmlDeviceGetCount()))
        else:
            visible_devices = {int(x.strip()) for x in visible_devices.split(',')}
        for i, real_id in enumerate(visible_devices):
            h = nvmlDeviceGetHandleByIndex(real_id)
            info = nvmlDeviceGetMemoryInfo(h)
            total = info.total
            free = info.free
            ratio = free / total
            gpus[i] = ratio
            # print(f'total    : {info.total}')
            # print(f'free     : {info.free}')
            # print(f'used     : {info.used}')
            # t = torch.cuda.get_device_properties(0).total_memory
            # c = torch.cuda.memory_cached(0)
            # a = torch.cuda.memory_allocated(0)
            # print(t, c, a)
        nvmlShutdown()
        return dict(sorted(gpus.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        logger.debug(f'Failed to get gpu info due to {e}')
        return dict((i, 1.0) for i in range(torch.cuda.device_count()))


def cuda_devices(query=None) -> List[int]:
    """Decide which GPUs to use

    Args:
      query:  (Default value = None)

    Returns:

    
    """
    if isinstance(query, list):
        if len(query) == 0:
            return [-1]
        return query
    if query is None:
        query = gpus_available()
        if not query:
            return []
        size, idx = max((v, k) for k, v in query.items())
        # When multiple GPUs have the same size, randomly pick one to avoid conflicting
        gpus_with_same_size = [k for k, v in query.items() if v == size]
        query = random.choice(gpus_with_same_size)
    if isinstance(query, float):
        gpus = gpus_available()
        if not query:
            return []
        query = [k for k, v in gpus.items() if v > query]
    elif isinstance(query, int):
        query = [query]
    return query


def pad_lists(sequences: List[List], dtype=torch.long, padding_value=0):
    return pad_sequence([torch.tensor(x, dtype=dtype) for x in sequences], True, padding_value)


def set_seed(seed=233, dont_care_speed=False):
    """Copied from https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/trainer.py#L76

    Args:
      seed:  (Default value = 233)
      dont_care_speed: True may have a negative single-run performance impact, but ensures deterministic

    Returns:

    
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # ^^ safe to call this function even if cuda is not available
    torch.cuda.manual_seed_all(seed)
    if dont_care_speed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def batched_index_select(input, index, dim=1):
    """

    Args:
      input: B x * x ... x *
      index: B x M
      dim:  (Default value = 1)

    Returns:

    
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def dtype_of(e: Union[int, bool, float]):
    if isinstance(e, bool):
        return torch.bool
    if isinstance(e, int):
        return torch.long
    if isinstance(e, float):
        return torch.float
    raise ValueError(f'Unsupported type of {repr(e)}')


def mean_model(model: torch.nn.Module):
    return float(torch.mean(torch.stack([torch.sum(p) for p in model.parameters() if p.requires_grad])))


def main():
    start = time.time()
    print(gpus_available())
    print(time.time() - start)
    # print(gpus_available())
    # print(cuda_devices())
    # print(cuda_devices(0.1))


if __name__ == '__main__':
    main()


def clip_grad_norm(model: nn.Module, grad_norm, transformer: nn.Module = None, transformer_grad_norm=None):
    if transformer_grad_norm is None:
        if grad_norm is not None:
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_norm)
    else:
        is_transformer = []
        non_transformer = []
        transformer = set(transformer.parameters())
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p in transformer:
                is_transformer.append(p)
            else:
                non_transformer.append(p)
        nn.utils.clip_grad_norm_(non_transformer, grad_norm)
        nn.utils.clip_grad_norm_(is_transformer, transformer_grad_norm)
