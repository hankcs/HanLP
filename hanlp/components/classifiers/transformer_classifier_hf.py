# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2023-02-17 17:54
import logging
from typing import List, Union, Callable

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizer, AutoTokenizer

from hanlp.common.dataset import TableDataset, PadSequenceDataLoader, SortingSamplerBuilder
from hanlp.common.torch_component import TorchComponent
from hanlp_common.constant import IDX
from hanlp_common.util import split_dict, reorder


class TransformerClassifierHF(TorchComponent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tokenizer: PreTrainedTokenizer = None

    def build_dataloader(self, data, sampler_builder=None, shuffle=False, device=None,
                         logger: logging.Logger = None,
                         **kwargs) -> DataLoader:
        dataset = TableDataset(data)
        lens = [len(sample['input_ids']) for sample in dataset]
        if sampler_builder:
            sampler = sampler_builder.build(lens, shuffle, 1)
        else:
            sampler = SortingSamplerBuilder(batch_size=32).build(lens, shuffle, 1)
        loader = PadSequenceDataLoader(dataset=dataset,
                                       batch_sampler=sampler,
                                       pad={'input_ids': self._tokenizer.pad_token_id},
                                       device=device,
                                       vocabs=self.vocabs)
        return loader

    def build_optimizer(self, **kwargs):
        raise NotImplementedError()

    def build_criterion(self, **kwargs):
        raise NotImplementedError()

    def build_metric(self, **kwargs):
        raise NotImplementedError()

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, **kwargs):
        raise NotImplementedError()

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        raise NotImplementedError()

    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, **kwargs):
        raise NotImplementedError()

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        self._tokenizer = AutoTokenizer.from_pretrained(save_dir)

    def load_weights(self, save_dir, filename='model.pt', **kwargs):
        pass

    def build_model(self, training=True, save_dir=None, **kwargs) -> torch.nn.Module:
        return AutoModelForSequenceClassification.from_pretrained(save_dir)

    def predict(self, text: Union[str, List[str]], topk=False, prob=False, **kwargs):
        """
        Classify text.

        Args:
            text: A document or a list of documents.
            topk: ``True`` or ``int`` to return the top-k labels.
            prob: Return also probabilities.
            max_len: Strip long document into ``max_len`` characters for faster prediction.
            **kwargs: Not used

        Returns:
            Classification results.
        """
        flat = isinstance(text, str)
        if flat:
            text = [text]
        if not isinstance(topk, list):
            topk = [topk] * len(text)
        if not isinstance(prob, list):
            prob = [prob] * len(text)
        # noinspection PyTypeChecker
        dataloader = self.build_dataloader(
            split_dict(self._tokenizer(text, max_length=self.model.config.max_position_embeddings, truncation=True,
                                       return_token_type_ids=False, return_attention_mask=False)),
            device=self.device)
        results = []
        order = []
        id2label = self.model.config.id2label
        for batch in dataloader:
            logits = self.model(input_ids=batch['input_ids']).logits
            logits, batch_labels = logits.sort(descending=True)
            batch_labels = [[id2label[l] for l in ls] for ls in batch_labels.tolist()]
            batch_probs = logits.softmax(dim=-1).tolist()
            for labels, probs, i in zip(batch_labels, batch_probs, batch[IDX]):
                k = topk[i]
                p = prob[i]
                if k is False:
                    labels = labels[0]
                elif k is True:
                    pass
                elif k:
                    labels = labels[:k]
                if p:
                    if k is False:
                        result = labels, probs[0]
                    else:
                        result = dict(zip(labels, probs))
                else:
                    result = labels
                results.append(result)
            order.extend(batch[IDX])
        results = reorder(results, order)
        if flat:
            results = results[0]
        return results

    @property
    def labels(self):
        return [x[1] for x in sorted(self.model.config.id2label.items())]
