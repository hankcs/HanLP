# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 20:27
import math
import os
import random
import tempfile
import warnings
from abc import ABC, abstractmethod
from copy import copy
from logging import Logger
from typing import Union, List, Callable, Iterable, Dict, Any

import torch
import torch.multiprocessing as mp
from hanlp.common.transform import TransformList, VocabDict, EmbeddingNamedTransform
from hanlp.common.vocab import Vocab
from hanlp.components.parsers.alg import kmeans
from hanlp.utils.io_util import read_cells, get_resource
from hanlp.utils.time_util import CountdownTimer
from hanlp.utils.torch_util import dtype_of
from hanlp_common.configurable import AutoConfigurable
from hanlp_common.constant import IDX, HANLP_VERBOSE
from hanlp_common.util import isdebugging, merge_list_of_dict, k_fold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataset import IterableDataset


class Transformable(ABC):
    def __init__(self, transform: Union[Callable, List] = None) -> None:
        """An object which can be transformed with a list of functions. It can be treated as an objected being passed
        through a list of functions, while these functions are kept in a list.

        Args:
            transform: A transform function or a list of functions.
        """
        super().__init__()
        if isinstance(transform, list) and not isinstance(transform, TransformList):
            transform = TransformList(*transform)
        self.transform: Union[Callable, TransformList] = transform

    def append_transform(self, transform: Callable):
        """Append a transform to its list of transforms.

        Args:
            transform: A new transform to be appended.

        Returns:
            Itself.

        """
        assert transform is not None, 'None transform not allowed'
        if not self.transform:
            self.transform = TransformList(transform)
        elif not isinstance(self.transform, TransformList):
            if self.transform != transform:
                self.transform = TransformList(self.transform, transform)
        else:
            if transform not in self.transform:
                self.transform.append(transform)
        return self

    def insert_transform(self, index: int, transform: Callable):
        """Insert a transform to a certain position.

        Args:
            index: A certain position.
            transform: A new transform.

        Returns:
            Itself.

        """
        assert transform is not None, 'None transform not allowed'
        if not self.transform:
            self.transform = TransformList(transform)
        elif not isinstance(self.transform, TransformList):
            if self.transform != transform:
                self.transform = TransformList(self.transform)
                self.transform.insert(index, transform)
        else:
            if transform not in self.transform:
                self.transform.insert(index, transform)
        return self

    def transform_sample(self, sample: dict, inplace=False) -> dict:
        """Apply transforms to a sample.

        Args:
            sample: A sample, which is a ``dict`` holding features.
            inplace: ``True`` to apply transforms inplace.

        .. Attention::
            If any transform modifies existing features, it will modify again and again when ``inplace=True``.
            For example, if a transform insert a ``BOS`` token to a list inplace, and it is called twice,
            then 2 ``BOS`` will be inserted which might not be an intended result.

        Returns:
            Transformed sample.
        """
        if not inplace:
            sample = copy(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample


class TransformableDataset(Transformable, Dataset, ABC):

    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None) -> None:
        """A :class:`~torch.utils.data.Dataset` which can be applied with a list of transform functions.

        Args:
            data: The local or remote path to a dataset, or a list of samples where each sample is a dict.
            transform: Predefined transform(s).
            cache: ``True`` to enable caching, so that transforms won't be called twice.
            generate_idx: Create a :const:`~hanlp_common.constants.IDX` field for each sample to store its order in dataset. Useful for prediction when
                samples are re-ordered by a sampler.
        """
        super().__init__(transform)
        if generate_idx is None:
            generate_idx = isinstance(data, list)
        data = self.load_data(data, generate_idx)
        assert data, 'No samples loaded'
        assert isinstance(data[0],
                          dict), f'TransformDataset expects each sample to be a dict but got {type(data[0])} instead.'
        self.data = data
        if cache:
            self.cache = [None] * len(data)
        else:
            self.cache = None

    def load_data(self, data, generate_idx=False):
        """A intermediate step between constructor and calling the actual file loading method.

        Args:
            data: If data is a file, this method calls :meth:`~hanlp.common.dataset.TransformableDataset.load_file`
                to load it.
            generate_idx: Create a :const:`~hanlp_common.constants.IDX` field for each sample to store its order in dataset. Useful for prediction when
                samples are re-ordered by a sampler.

        Returns: Loaded samples.

        """
        if self.should_load_file(data):
            if isinstance(data, str):
                data = get_resource(data)
            data = list(self.load_file(data))
        if generate_idx:
            for i, each in enumerate(data):
                each[IDX] = i
        # elif isinstance(data, list):
        #     data = self.load_list(data)
        return data

    # noinspection PyMethodMayBeStatic
    # def load_list(self, data: list) -> List[Dict[str, Any]]:
    #     return data

    def should_load_file(self, data) -> bool:
        """Determines whether data is a filepath.

        Args:
            data: Data to check.

        Returns: ``True`` to indicate it's a filepath.

        """
        return isinstance(data, str)

    @abstractmethod
    def load_file(self, filepath: str):
        """The actual file loading logic.

        Args:
            filepath: The path to a dataset.
        """
        pass

    def __getitem__(self, index: Union[int, slice]) -> Union[dict, List[dict]]:
        """ Get the index-th sample in this dataset.

        Args:
            index: Either a integer index of a list of indices.

        Returns: Either a sample or or list of samples depending on how many indices are passed in.

        """
        # if isinstance(index, (list, tuple)):
        #     assert len(index) == 1
        #     index = index[0]
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self[i] for i in indices]

        if self.cache:
            cache = self.cache[index]
            if cache:
                return cache
        sample = self.data[index]
        sample = self.transform_sample(sample)
        if self.cache:
            self.cache[index] = sample
        return sample

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f'{len(self)} samples: {self[0]} ...'

    def purge_cache(self):
        """Purges all cache. If cache is not enabled, this method enables it.
        """
        self.cache = [None] * len(self.data)

    def split(self, *ratios):
        """Split dataset into subsets.

        Args:
            *ratios: The ratios for each subset. They can be any type of numbers which will be normalized. For example,
                    ``8, 1, 1`` are equivalent to ``0.8, 0.1, 0.1``.

        Returns:
            list[TransformableDataset]: A list of subsets.
        """
        ratios = [x / sum(ratios) for x in ratios]
        chunks = []
        prev = 0
        for r in ratios:
            cur = prev + math.ceil(len(self) * r)
            chunks.append([prev, cur])
            prev = cur
        chunks[-1][1] = len(self)
        outputs = []
        for b, e in chunks:
            dataset = copy(self)
            dataset.data = dataset.data[b:e]
            if dataset.cache:
                dataset.cache = dataset.cache[b:e]
            outputs.append(dataset)
        return outputs

    def k_fold(self, k, i):
        """Perform k-fold sampling.

        Args:
            k (int): Number of folds.
            i (int): The i-th fold.

        Returns:
            TransformableDataset: The i-th fold subset of this dataset.

        """
        assert 0 <= i <= k, f'Invalid split {i}'
        train_indices, test_indices = k_fold(k, len(self), i)
        return self.subset(train_indices), self.subset(test_indices)

    def subset(self, indices):
        """Create a subset given indices of samples.

        Args:
            indices: Indices of samples.

        Returns:
            TransformableDataset: The a subset of this dataset.
        """
        dataset = copy(self)
        dataset.data = [dataset.data[i] for i in indices]
        if dataset.cache:
            dataset.cache = [dataset.cache[i] for i in indices]
        return dataset

    def shuffle(self):
        """Shuffle this dataset inplace.
        """
        if not self.cache:
            random.shuffle(self.data)
        else:
            z = list(zip(self.data, self.cache))
            random.shuffle(z)
            self.data, self.cache = zip(*z)

    def prune(self, criterion: Callable, logger: Logger = None):
        """Prune (to discard) samples according to a criterion.

        Args:
            criterion: A functions takes a sample as input and output ``True`` if the sample needs to be pruned.
            logger: If any, log statistical messages using it.

        Returns:
            int: Size before pruning.
        """
        # noinspection PyTypeChecker
        size_before = len(self)
        good_ones = [i for i, s in enumerate(self) if not criterion(s)]
        self.data = [self.data[i] for i in good_ones]
        if self.cache:
            self.cache = [self.cache[i] for i in good_ones]
        if logger:
            size_after = len(self)
            num_pruned = size_before - size_after
            logger.info(f'Pruned [yellow]{num_pruned} ({num_pruned / size_before:.1%})[/yellow] '
                        f'samples out of {size_before}.')
        return size_before


class TransformSequentialDataset(Transformable, IterableDataset, ABC):
    pass


class DeviceDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=None, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 device=None, **kwargs):
        if batch_sampler is not None:
            batch_size = 1
        if num_workers is None:
            if isdebugging():
                num_workers = 0
            else:
                num_workers = 2
        # noinspection PyArgumentList
        super(DeviceDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                               sampler=sampler,
                                               batch_sampler=batch_sampler, num_workers=num_workers,
                                               collate_fn=collate_fn,
                                               pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                               worker_init_fn=worker_init_fn,
                                               multiprocessing_context=multiprocessing_context, **kwargs)
        self.device = device

    def __iter__(self):
        for raw_batch in super(DeviceDataLoader, self).__iter__():
            if self.device is not None:
                for field, data in raw_batch.items():
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device)
                        raw_batch[field] = data
            yield raw_batch

    def collate_fn(self, samples):
        return merge_list_of_dict(samples)


class PadSequenceDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=32, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 pad: dict = None, vocabs: VocabDict = None, device=None, **kwargs):
        """ A dataloader commonly used for NLP tasks. It offers the following convenience.

        - Bachify each field of samples into a :class:`~torch.Tensor` if the field name satisfies the following criterion.
            - Name ends with _id, _ids, _count, _offset, _span, mask
            - Name is in `pad` dict.

        - Pad each field according to field name, the vocabs and pad dict.
        - Move :class:`~torch.Tensor` onto device.

        Args:
            dataset: A :class:`~torch.utils.data.Dataset` to be bachified.
            batch_size: Max size of each batch.
            shuffle: ``True`` to shuffle batches.
            sampler: A :class:`~torch.utils.data.Sampler` to sample samples from data.
            batch_sampler: A :class:`~torch.utils.data.Sampler` to sample batches form all batches.
            num_workers: Number of workers for multi-thread loading. Note that multi-thread loading aren't always
                faster.
            collate_fn: A function to perform batchifying. It must be set to ``None`` in order to make use of the
                 features this class offers.
            pin_memory: If samples are loaded in the Dataset on CPU and would like to be pushed to
                    the GPU, enabling pin_memory can speed up the transfer. It's not useful since most data field are
                    not in Tensor type.
            drop_last: Drop the last batch since it could be half-empty.
            timeout: For multi-worker loading, set a timeout to wait for a worker.
            worker_init_fn: Init function for multi-worker.
            multiprocessing_context: Context for multiprocessing.
            pad: A dict holding field names and their padding values.
            vocabs: A dict of vocabs so padding value can be fetched from it.
            device: The device tensors will be moved onto.
            **kwargs: Other arguments will be passed to :meth:`torch.utils.data.Dataset.__init__`
        """
        if device == -1:
            device = None
        if collate_fn is None:
            collate_fn = self.collate_fn
        if num_workers is None:
            if isdebugging():
                num_workers = 0
            else:
                num_workers = 2
        if batch_sampler is None:
            assert batch_size, 'batch_size has to be specified when batch_sampler is None'
        else:
            batch_size = 1
            shuffle = None
            drop_last = None
        # noinspection PyArgumentList
        super(PadSequenceDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                    sampler=sampler,
                                                    batch_sampler=batch_sampler, num_workers=num_workers,
                                                    collate_fn=collate_fn,
                                                    pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                                    worker_init_fn=worker_init_fn,
                                                    multiprocessing_context=multiprocessing_context, **kwargs)
        self.vocabs = vocabs
        if isinstance(dataset, TransformableDataset) and dataset.transform:
            transform = dataset.transform
            if not isinstance(transform, TransformList):
                transform = []
            for each in transform:
                if isinstance(each, EmbeddingNamedTransform):
                    if pad is None:
                        pad = {}
                    if each.dst not in pad:
                        pad[each.dst] = 0
        self.pad = pad
        self.device = device

    def __iter__(self):
        for raw_batch in super(PadSequenceDataLoader, self).__iter__():
            yield self.tensorize(raw_batch, vocabs=self.vocabs, pad_dict=self.pad, device=self.device)

    @staticmethod
    def tensorize(raw_batch: Dict[str, Any], vocabs: VocabDict, pad_dict: Dict[str, int] = None, device=None):
        for field, data in raw_batch.items():
            if isinstance(data, torch.Tensor):
                continue
            vocab_key = field[:-len('_id')] if field.endswith('_id') else None
            vocab: Vocab = vocabs.get(vocab_key, None) if vocabs and vocab_key else None
            if vocab:
                pad = vocab.safe_pad_token_idx
                dtype = torch.long
            elif pad_dict is not None and field in pad_dict:
                pad = pad_dict[field]
                dtype = dtype_of(pad)
            elif field.endswith('_offset') or field.endswith('_id') or field.endswith(
                    '_count') or field.endswith('_ids') or field.endswith('_score') or field.endswith(
                '_length') or field.endswith('_span'):
                # guess some common fields to pad
                pad = 0
                dtype = torch.long
            elif field.endswith('_mask'):
                pad = False
                dtype = torch.bool
            else:
                # no need to pad
                continue
            data = PadSequenceDataLoader.pad_data(data, pad, dtype)
            raw_batch[field] = data
        if device is not None:
            for field, data in raw_batch.items():
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                    raw_batch[field] = data
        return raw_batch

    @staticmethod
    def pad_data(data: Union[torch.Tensor, Iterable], pad, dtype=None, device=None):
        """Perform the actual padding for a given data.

        Args:
            data: Data to be padded.
            pad: Padding value.
            dtype: Data type.
            device: Device to be moved onto.

        Returns:
            torch.Tensor: A ``torch.Tensor``.
        """
        if isinstance(data[0], torch.Tensor):
            data = pad_sequence(data, True, pad)
        elif isinstance(data[0], Iterable):
            inner_is_iterable = False
            for each in data:
                if len(each):
                    if isinstance(each[0], Iterable):
                        inner_is_iterable = True
                        if len(each[0]):
                            if not dtype:
                                dtype = dtype_of(each[0][0])
                    else:
                        inner_is_iterable = False
                        if not dtype:
                            dtype = dtype_of(each[0])
                    break
            if inner_is_iterable:
                max_seq_len = len(max(data, key=len))
                max_word_len = len(max([chars for words in data for chars in words], key=len))
                ids = torch.zeros(len(data), max_seq_len, max_word_len, dtype=dtype, device=device)
                for i, words in enumerate(data):
                    for j, chars in enumerate(words):
                        ids[i][j][:len(chars)] = torch.tensor(chars, dtype=dtype, device=device)
                data = ids
            else:
                data = pad_sequence([torch.tensor(x, dtype=dtype, device=device) for x in data], True, pad)
        elif isinstance(data, list):
            data = torch.tensor(data, dtype=dtype, device=device)
        return data

    def collate_fn(self, samples):
        return merge_list_of_dict(samples)


class CachedDataLoader(object):
    def __init__(self, dataloader: torch.utils.data.DataLoader, filename=None):
        if not filename:
            filename = tempfile.NamedTemporaryFile(prefix='hanlp-cache-', delete=False).name
        self.filename = filename
        self.size = len(dataloader)
        self._build_cache(dataloader)

    def _build_cache(self, dataset, verbose=HANLP_VERBOSE):
        timer = CountdownTimer(self.size)
        with open(self.filename, "wb") as f:
            for i, batch in enumerate(dataset):
                torch.save(batch, f, _use_new_zipfile_serialization=False)
                if verbose:
                    timer.log(f'Caching {self.filename} [blink][yellow]...[/yellow][/blink]')

    def close(self):
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def __iter__(self):
        with open(self.filename, "rb") as f:
            for i in range(self.size):
                batch = torch.load(f)
                yield batch

    def __len__(self):
        return self.size


def _prefetch_generator(dataloader, queue, batchify=None):
    while True:
        for batch in dataloader:
            if batchify:
                batch = batchify(batch)
            queue.put(batch)


class PrefetchDataLoader(DataLoader):
    def __init__(self, dataloader: torch.utils.data.DataLoader, prefetch: int = 10, batchify: Callable = None) -> None:
        """ A dataloader wrapper which speeds up bachifying using multi-processing. It works best for dataloaders
        of which the bachify takes very long time. But it introduces extra GPU memory consumption since prefetched
        batches are stored in a ``Queue`` on GPU.

        .. Caution::

            PrefetchDataLoader only works in spawn mode with the following initialization code:

            Examples::

                if __name__ == '__main__':
                    import torch

                    torch.multiprocessing.set_start_method('spawn')

            And these 2 lines **MUST** be put into ``if __name__ == '__main__':`` block.

        Args:
            dataloader: A :class:`~torch.utils.data.DatasetLoader` to be prefetched.
            prefetch: Number of batches to prefetch.
            batchify: A bachify function called on each batch of samples. In which case, the inner dataloader shall
                    return samples without really bachify them.
        """
        super().__init__(dataset=dataloader)
        self._batchify = batchify
        self.prefetch = None if isdebugging() else prefetch
        if self.prefetch:
            self._fire_process(dataloader, prefetch)

    def _fire_process(self, dataloader, prefetch):
        self.queue = mp.Queue(prefetch)
        self.process = mp.Process(target=_prefetch_generator, args=(dataloader, self.queue, self._batchify))
        self.process.start()

    def __iter__(self):
        if not self.prefetch:
            for batch in self.dataset:
                if self._batchify:
                    batch = self._batchify(batch)
                yield batch
        else:
            size = len(self)
            while size:
                batch = self.queue.get()
                yield batch
                size -= 1

    def close(self):
        """Close this dataloader and terminates internal processes and queue. It's recommended to call this method to
            ensure a program can gracefully shutdown.
        """
        if self.prefetch:
            self.queue.close()
            self.process.terminate()

    @property
    def batchify(self):
        return self._batchify

    @batchify.setter
    def batchify(self, batchify):
        self._batchify = batchify
        if not self.prefetch:
            prefetch = vars(self.queue).get('maxsize', 10)
            self.close()
            self._fire_process(self.dataset, prefetch)


class BucketSampler(Sampler):
    # noinspection PyMissingConstructor
    def __init__(self, buckets: Dict[float, List[int]], batch_max_tokens, batch_size=None, shuffle=False):
        """A bucketing based sampler which groups samples into buckets then creates batches from each bucket.

        Args:
            buckets: A dict of which keys are some statistical numbers of each bucket, and values are the indices of
                samples in each bucket.
            batch_max_tokens: Maximum tokens per batch.
            batch_size: Maximum samples per batch.
            shuffle: ``True`` to shuffle batches and samples in a batch.
        """
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[
            (size, bucket) for size, bucket in buckets.items()
        ])
        # the number of chunks in each bucket, which is clipped by
        # range [1, len(bucket)]
        if batch_size:
            self.chunks = [
                max(batch_size, min(len(bucket), max(round(size * len(bucket) / batch_max_tokens), 1)))
                for size, bucket in zip(self.sizes, self.buckets)
            ]
        else:
            self.chunks = [
                min(len(bucket), max(round(size * len(bucket) / batch_max_tokens), 1))
                for size, bucket in zip(self.sizes, self.buckets)
            ]

    def __iter__(self):
        # if shuffle, shuffle both the buckets and samples in each bucket
        range_fn = torch.randperm if self.shuffle else torch.arange
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1 for j in range(self.chunks[i])]
            # DON'T use `torch.chunk` which may return wrong number of chunks
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        return sum(self.chunks)


class KMeansSampler(BucketSampler):
    def __init__(self, lengths, batch_max_tokens, batch_size=None, shuffle=False, n_buckets=1):
        """A bucket sampler which groups samples using KMeans on their lengths.

        Args:
            lengths: Lengths of each sample, usually measured by number of tokens.
            batch_max_tokens: Maximum tokens per batch.
            batch_size: Maximum samples per batch.
            shuffle: ``True`` to shuffle batches. Samples in the same batch won't be shuffled since the ordered sequence
                    is helpful to speed up RNNs.
            n_buckets: Number of buckets. Clusters in terms of KMeans.
        """
        if n_buckets > len(lengths):
            n_buckets = 1
        self.n_buckets = n_buckets
        self.lengths = lengths
        buckets = dict(zip(*kmeans(self.lengths, n_buckets)))
        super().__init__(buckets, batch_max_tokens, batch_size, shuffle)


class SortingSampler(Sampler):
    # noinspection PyMissingConstructor
    def __init__(self, lengths: List[int], batch_size=None, batch_max_tokens=None, shuffle=False) -> None:
        """A sampler which sort samples according to their lengths. It takes a continuous chunk of sorted samples to
        make a batch.

        Args:
            lengths: Lengths of each sample, usually measured by number of tokens.
            batch_max_tokens: Maximum tokens per batch.
            batch_size: Maximum samples per batch.
            shuffle: ``True`` to shuffle batches and samples in a batch.
        """
        # assert any([batch_size, batch_max_tokens]), 'At least one of batch_size and batch_max_tokens is required'
        self.shuffle = shuffle
        self.batch_size = batch_size
        # self.batch_max_tokens = batch_max_tokens
        self.batch_indices = []
        num_tokens = 0
        mini_batch = []
        for i in torch.argsort(torch.tensor(lengths), descending=True).tolist():
            # if batch_max_tokens:
            if (batch_max_tokens is None or num_tokens + lengths[i] <= batch_max_tokens) and (
                    batch_size is None or len(mini_batch) < batch_size):
                mini_batch.append(i)
                num_tokens += lengths[i]
            else:
                if not mini_batch:  # this sequence is longer than  batch_max_tokens
                    mini_batch.append(i)
                    self.batch_indices.append(mini_batch)
                    mini_batch = []
                    num_tokens = 0
                else:
                    self.batch_indices.append(mini_batch)
                    mini_batch = [i]
                    num_tokens = lengths[i]
        if mini_batch:
            self.batch_indices.append(mini_batch)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batch_indices)
        for batch in self.batch_indices:
            yield batch

    def __len__(self) -> int:
        return len(self.batch_indices)


class SamplerBuilder(AutoConfigurable, ABC):
    @abstractmethod
    def build(self, lengths: List[int], shuffle=False, gradient_accumulation=1, **kwargs) -> Sampler:
        """Build a ``Sampler`` given statistics of samples and other arguments.

        Args:
            lengths: The lengths of samples.
            shuffle: ``True`` to shuffle batches. Note samples in each mini-batch are not necessarily shuffled.
            gradient_accumulation: Number of mini-batches per update step.
            **kwargs: Other arguments to be passed to the constructor of the sampler.
        """
        pass

    def __call__(self, lengths: List[int], shuffle=False, **kwargs) -> Sampler:
        return self.build(lengths, shuffle, **kwargs)

    def scale(self, gradient_accumulation):
        r"""Scale down the ``batch_size`` and ``batch_max_tokens`` to :math:`\frac{1}{\text{gradient_accumulation}}`
        of them respectively.

        Args:
            gradient_accumulation: Number of mini-batches per update step.

        Returns:
            tuple(int,int): batch_size, batch_max_tokens
        """
        batch_size = self.batch_size
        batch_max_tokens = self.batch_max_tokens
        if gradient_accumulation:
            if batch_size:
                batch_size //= gradient_accumulation
            if batch_max_tokens:
                batch_max_tokens //= gradient_accumulation
        return batch_size, batch_max_tokens


class SortingSamplerBuilder(SortingSampler, SamplerBuilder):
    # noinspection PyMissingConstructor
    def __init__(self, batch_size=None, batch_max_tokens=None) -> None:
        """Builds a :class:`~hanlp.common.dataset.SortingSampler`.

        Args:
            batch_max_tokens: Maximum tokens per batch.
            batch_size: Maximum samples per batch.
        """
        self.batch_max_tokens = batch_max_tokens
        self.batch_size = batch_size

    def build(self, lengths: List[int], shuffle=False, gradient_accumulation=1, **kwargs) -> Sampler:
        batch_size, batch_max_tokens = self.scale(gradient_accumulation)
        return SortingSampler(lengths, batch_size, batch_max_tokens, shuffle)

    def __len__(self) -> int:
        return 1


class KMeansSamplerBuilder(KMeansSampler, SamplerBuilder):
    # noinspection PyMissingConstructor
    def __init__(self, batch_max_tokens, batch_size=None, n_buckets=1):
        """Builds a :class:`~hanlp.common.dataset.KMeansSampler`.

        Args:
            batch_max_tokens: Maximum tokens per batch.
            batch_size: Maximum samples per batch.
            n_buckets: Number of buckets. Clusters in terms of KMeans.
        """
        self.n_buckets = n_buckets
        self.batch_size = batch_size
        self.batch_max_tokens = batch_max_tokens

    def build(self, lengths: List[int], shuffle=False, gradient_accumulation=1, **kwargs) -> Sampler:
        batch_size, batch_max_tokens = self.scale(gradient_accumulation)
        return KMeansSampler(lengths, batch_max_tokens, batch_size, shuffle, self.n_buckets)

    def __len__(self) -> int:
        return 1


class TableDataset(TransformableDataset):
    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 delimiter='auto',
                 strip=True,
                 headers=None) -> None:
        self.headers = headers
        self.strip = strip
        self.delimiter = delimiter
        super().__init__(data, transform, cache)

    def load_file(self, filepath: str):
        for idx, cells in enumerate(read_cells(filepath, strip=self.strip, delimiter=self.delimiter)):
            if not idx and not self.headers:
                self.headers = cells
                if any(len(h) > 32 for h in self.headers):
                    warnings.warn('As you did not pass in `headers` to `TableDataset`, the first line is regarded as '
                                  'headers. However, the length for some headers are too long (>32), which might be '
                                  'wrong. To make sure, pass `headers=...` explicitly.')
            else:
                yield dict(zip(self.headers, cells))
