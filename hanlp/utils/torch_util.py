# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 15:52
import os
import random
import time
from typing import List, Union, Dict, Tuple

import numpy as np
import torch
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit, nvmlShutdown, nvmlDeviceGetCount
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from hanlp.utils.io_util import get_resource, replace_ext, TimingFileIterator
from hanlp.utils.log_util import logger, flash
from hanlp_common.constant import HANLP_VERBOSE
from hanlp_common.io import load_pickle, save_pickle


def gpus_available() -> Dict[int, float]:
    if not torch.cuda.is_available():
        return dict()
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


def load_word2vec(path, delimiter=' ', cache=True) -> Tuple[Dict[str, np.ndarray], int]:
    realpath = get_resource(path)
    binpath = replace_ext(realpath, '.pkl')
    if cache:
        try:
            flash('Loading word2vec from cache [blink][yellow]...[/yellow][/blink]')
            word2vec, dim = load_pickle(binpath)
            flash('')
            return word2vec, dim
        except IOError:
            pass

    dim = None
    word2vec = dict()
    f = TimingFileIterator(realpath)
    for idx, line in enumerate(f):
        f.log('Loading word2vec from text file [blink][yellow]...[/yellow][/blink]')
        line = line.rstrip().split(delimiter)
        if len(line) > 2:
            if dim is None:
                dim = len(line)
            else:
                if len(line) != dim:
                    logger.warning('{}#{} length mismatches with {}'.format(path, idx + 1, dim))
                    continue
            word, vec = line[0], line[1:]
            word2vec[word] = np.array(vec, dtype=np.float32)
    dim -= 1
    if cache:
        flash('Caching word2vec [blink][yellow]...[/yellow][/blink]')
        save_pickle((word2vec, dim), binpath)
        flash('')
    return word2vec, dim


def load_word2vec_as_vocab_tensor(path, delimiter=' ', cache=True) -> Tuple[Dict[str, int], torch.Tensor]:
    realpath = get_resource(path)
    vocab_path = replace_ext(realpath, '.vocab')
    matrix_path = replace_ext(realpath, '.pt')
    if cache:
        try:
            if HANLP_VERBOSE:
                flash('Loading vocab and matrix from cache [blink][yellow]...[/yellow][/blink]')
            vocab = load_pickle(vocab_path)
            matrix = torch.load(matrix_path, map_location='cpu')
            if HANLP_VERBOSE:
                flash('')
            return vocab, matrix
        except IOError:
            pass

    word2vec, dim = load_word2vec(path, delimiter, cache)
    vocab = dict((k, i) for i, k in enumerate(word2vec.keys()))
    matrix = torch.Tensor(np.stack(list(word2vec.values())))
    if cache:
        flash('Caching vocab and matrix [blink][yellow]...[/yellow][/blink]')
        save_pickle(vocab, vocab_path)
        torch.save(matrix, matrix_path)
        flash('')
    return vocab, matrix


def save_word2vec(word2vec: dict, filepath, delimiter=' '):
    with open(filepath, 'w', encoding='utf-8') as out:
        for w, v in word2vec.items():
            out.write(f'{w}{delimiter}')
            out.write(f'{delimiter.join(str(x) for x in v)}\n')


def lengths_to_mask(seq_len, max_len=None):
    r"""
    .. code-block::

        >>> seq_len = torch.arange(2, 16)
        >>> mask = lengths_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = lengths_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = lengths_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param torch.LongTensor seq_len: (B,)
    :param int max_len: max sequence length。
    :return:  torch.Tensor  (B, max_len)
    """
    assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
    batch_size = seq_len.size(0)
    max_len = int(max_len) if max_len else seq_len.max().long()
    broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
    mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))

    return mask


def activation_from_name(name: str):
    return getattr(torch.nn, name)