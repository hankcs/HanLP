# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-02 16:51
import logging
import os
import warnings
from abc import ABC, abstractmethod
from copy import copy
from typing import Callable, Dict, Any, Union, Iterable, List

import torch
from hanlp_common.util import merge_locals_kwargs
from torch.utils.data import DataLoader

from hanlp_common.constant import BOS, EOS
from hanlp.common.dataset import SamplerBuilder, SortingSamplerBuilder, TransformableDataset, KMeansSamplerBuilder
from hanlp_common.document import Document
from hanlp.common.structure import ConfigTracker
from hanlp.common.torch_component import TorchComponent
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp.utils.time_util import CountdownTimer


class Task(ConfigTracker, TorchComponent, ABC):
    # noinspection PyMissingConstructor
    def __init__(self,
                 trn: str = None,
                 dev: str = None,
                 tst: str = None,
                 sampler_builder: SamplerBuilder = None,
                 dependencies: str = None,
                 scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=False,
                 lr=None,
                 separate_optimizer=False,
                 cls_is_bos=False,
                 sep_is_eos=False,
                 **kwargs) -> None:
        """
        A task in the multi-task learning framework

        Args:
            trn: Path to training set.
            dev: Path to dev set.
            tst: Path to test set.
            sampler_builder: A builder which builds a sampler.
            dependencies: Its dependencies on other tasks.
            scalar_mix: A builder which builds a `ScalarMixWithDropout` object.
            use_raw_hidden_states: Whether to use raw hidden states from transformer without any pooling.
            lr: Learning rate for this task.
            separate_optimizer: Use customized separate optimizer for this task.
            cls_is_bos: ``True`` to treat the first token as ``BOS``.
            sep_is_eos: ``True`` to treat the last token as ``EOS``.
            **kwargs: Additional config.
        """
        ConfigTracker.__init__(self, merge_locals_kwargs(locals(), kwargs))
        for f, n in zip([trn, dev, tst], ['trn', 'dev', 'tst']):
            if f and os.path.isfile(f):  # anonymize local file names
                self.config.pop(n)
        self.separate_optimizer = separate_optimizer
        self.lr = lr
        self.use_raw_hidden_states = use_raw_hidden_states
        if sampler_builder is None:
            sampler_builder = SortingSamplerBuilder(batch_size=32)
        self.sampler_builder: Union[SortingSamplerBuilder, KMeansSamplerBuilder] = sampler_builder
        self.dependencies = dependencies
        self.tst = tst
        self.dev = dev
        self.trn = trn
        self.scalar_mix = scalar_mix
        self.cls_is_bos = cls_is_bos
        self.sep_is_eos = sep_is_eos

    @abstractmethod
    def build_dataloader(self,
                         data,
                         transform: Callable = None,
                         training=False,
                         device=None,
                         logger: logging.Logger = None,
                         cache=False,
                         gradient_accumulation=1,
                         **kwargs) -> DataLoader:
        """
        Build a dataloader for training or evaluation.

        Args:
            data: Either a path or a list of samples.
            transform: The transform from MTL, which is usually [TransformerSequenceTokenizer, FieldLength('token')]
            training: Whether this method is called on training set.
            device: The device dataloader is intended to work with.
            logger: Logger for printing message indicating progress.
            cache: Whether the dataloader should be cached.
            gradient_accumulation: Gradient accumulation to be passed to sampler builder.
            **kwargs: Additional experimental arguments.
        """
        pass

    def build_optimizer(self, decoder: torch.nn.Module, **kwargs):
        pass

    def build_batch_wise_scheduler(self, decoder: torch.nn.Module, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion,
                     ) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        pass

    @abstractmethod
    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any], decoder: torch.nn.Module, **kwargs) -> Union[Dict[str, Any], Any]:
        pass

    @abstractmethod
    def update_metrics(self,
                       batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any],
                       metric: Union[MetricDict, Metric]):
        pass

    # noinspection PyMethodOverriding
    @abstractmethod
    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        pass

    @abstractmethod
    def build_metric(self, **kwargs):
        pass

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        pass

    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, output=False, **kwargs):
        pass

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, **kwargs):
        pass

    # noinspection PyMethodMayBeStatic
    def compute_lens(self, data: Union[List[Dict[str, Any]], str], dataset: TransformableDataset,
                     input_ids='token_input_ids', length_field='token'):
        """

        Args:
            data: Samples to be measured or path to dataset during training time.
            dataset: During training time, use this dataset to measure the length of each sample inside.
            input_ids: Field name corresponds to input ids.
            length_field: Fall back to this field during prediction as input_ids may not be generated yet.

        Returns:

            Length list of this samples

        """
        if isinstance(data, str):
            if not dataset.cache:
                warnings.warn(f'Caching for the dataset is not enabled, '
                              f'try `dataset.purge_cache()` if possible. The dataset is {dataset}.')
            timer = CountdownTimer(len(dataset))
            for each in dataset:
                timer.log('Preprocessing and caching samples [blink][yellow]...[/yellow][/blink]')
            timer.erase()
            return [len(x[input_ids]) for x in dataset]
        return [len(x[length_field]) for x in data]

    def feed_batch(self,
                   h: torch.FloatTensor,
                   batch: Dict[str, torch.Tensor],
                   mask: torch.BoolTensor,
                   decoder: torch.nn.Module):
        return decoder(h, batch=batch, mask=mask)

    def input_is_flat(self, data) -> bool:
        """
        Check whether the data is flat (meaning that it's only a single sample, not even batched).

        Returns:
            bool: ``True`` to indicate the input data is flat.
        """
        raise NotImplementedError(
            '`input_is_flat()` needs to be implemented for the task component to accept raw input from user.'
        )

    @abstractmethod
    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def transform_batch(self,
                        batch: Dict[str, Any],
                        # inputs: List[List[str]],
                        results: Dict[str, Any] = None,
                        cls_is_bos=False,
                        sep_is_eos=False) -> Dict[str, Any]:
        """
        Let the task transform the batch before feeding the batch into its decoder. The default behavior is to
        adjust the head and tail of tokens, according to ``cls_is_bos``, ``sep_is_eos`` passed in and the two
        settings of the task itself.

        Args:
            batch: A batch of samples.
            results: Predicted results from other tasks which might be useful for this task to utilize. Say a dep task
                uses both token and pos as features, then it will need both tok and pos results to make a batch.
            cls_is_bos: First token in this batch is BOS.
            sep_is_eos: Last token in this batch is EOS.

        Returns:
            A batch.

        """
        if cls_is_bos != self.cls_is_bos or sep_is_eos != self.sep_is_eos:
            batch = copy(batch)
            tokens = self._adjust_token(batch, cls_is_bos, sep_is_eos, 'token')
            delta = len(tokens[0]) - len(batch['token'][0])
            batch['token_length'] = batch['token_length'] + delta
            batch['token'] = tokens
            if 'token_' in batch:
                if isinstance(batch['token_'][0], list):
                    batch['token_'] = self._adjust_token(batch, cls_is_bos, sep_is_eos, 'token_')
                else:
                    batch['token_'] = tokens
        return batch

    def _adjust_token(self, batch, cls_is_bos, sep_is_eos, token_key):
        tokens = []
        for sent in batch[token_key]:
            if cls_is_bos:
                if not self.cls_is_bos:
                    sent = sent[1:]
            elif self.cls_is_bos:
                sent = [BOS] + sent
            if sep_is_eos:
                if not self.sep_is_eos:
                    sent = sent[:-1]
            elif self.sep_is_eos:
                sent = sent + [EOS]
            tokens.append(sent)
        return tokens

    # noinspection PyMethodMayBeStatic
    def build_samples(self, inputs, cls_is_bos=False, sep_is_eos=False):
        """
        Build samples for this task. Called when this task is the first task. Default behaviour is to take inputs as
        list of tokens and put these tokens into a dict per sample.

        Args:
            inputs: Inputs from users, usually a list of lists of tokens.
            cls_is_bos: Insert BOS to the head of each sentence.
            sep_is_eos: Append EOS to the tail of each sentence.

        Returns:
            List of samples.

        """
        if cls_is_bos:
            inputs = [[BOS] + x for x in inputs]
        if sep_is_eos:
            inputs = [x + [EOS] for x in inputs]
        return [{'token': token} for token in inputs]

    def build_tokenizer(self, tokenizer: TransformerSequenceTokenizer):
        """Build a transformer tokenizer for this task.

        Args:
            tokenizer: A tokenizer which is shared but can be adjusted to provide per-task settings.

        Returns:
            A TransformerSequenceTokenizer.

        """
        if tokenizer.cls_is_bos != self.cls_is_bos or tokenizer.sep_is_eos != self.sep_is_eos:
            tokenizer = copy(tokenizer)
            tokenizer.cls_is_bos = self.cls_is_bos
            tokenizer.sep_is_eos = self.sep_is_eos
        return tokenizer

    # noinspection PyMethodMayBeStatic
    def finalize_document(self, doc: Document, task_name: str):
        pass
