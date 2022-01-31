# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-01-29 21:07
import logging
import math
from typing import Callable, Union, List

import torch
from hanlp_common.constant import IDX
from hanlp_common.util import reorder
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer

from hanlp.common.dataset import TransformableDataset, PadSequenceDataLoader, SortingSampler
from hanlp.common.torch_component import TorchComponent
from hanlp.layers.transformers.pt_imports import AutoTokenizer_
from hanlp.transform.transformer_tokenizer import TransformerTextTokenizer
from hanlp.utils.time_util import CountdownTimer


class MaskedLanguageModelDataset(TransformableDataset):

    def load_file(self, filepath: str):
        raise NotImplementedError()


class MaskedLanguageModel(TorchComponent):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer: PreTrainedTokenizer = None

    def build_dataloader(self, data, batch_size, shuffle=False, device=None, logger: logging.Logger = None,
                         verbose=False, **kwargs) -> DataLoader:
        dataset = MaskedLanguageModelDataset([{'token': x} for x in data], generate_idx=True,
                                             transform=TransformerTextTokenizer(self.tokenizer, text_a_key='token'))
        if verbose:
            verbose = CountdownTimer(len(dataset))
        lens = []
        for each in dataset:
            lens.append(len(each['token_input_ids']))
            if verbose:
                verbose.log('Preprocessing and caching samples [blink][yellow]...[/yellow][/blink]')
        dataloader = PadSequenceDataLoader(dataset, batch_sampler=SortingSampler(lens, batch_size=batch_size),
                                           device=device)
        return dataloader

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

    def build_model(self, training=True, transformer=None, **kwargs) -> torch.nn.Module:
        return AutoModelForMaskedLM.from_pretrained(transformer)

    def input_is_flat(self, masked_sents):
        return isinstance(masked_sents, str)

    def predict(self, masked_sents: Union[str, List[str]], batch_size=32, topk=10, **kwargs):
        flat = self.input_is_flat(masked_sents)
        if flat:
            masked_sents = [masked_sents]
        dataloader = self.build_dataloader(masked_sents, **self.config, device=self.device, batch_size=batch_size)
        orders = []
        results = []
        for batch in dataloader:
            input_ids = batch['token_input_ids']
            outputs = self.model(input_ids=input_ids, attention_mask=batch['token_attention_mask'])
            mask = input_ids == self.tokenizer.mask_token_id
            if mask.any():
                num_masks = mask.sum(dim=-1).tolist()
                masked_logits = outputs.logits[mask]
                masked_logits[:, self.tokenizer.all_special_ids] = -math.inf
                probs, indices = torch.nn.functional.softmax(masked_logits, dim=-1).topk(topk)
                br = []
                for p, index in zip(probs.tolist(), indices.tolist()):
                    br.append(dict(zip(self.tokenizer.convert_ids_to_tokens(index), p)))
                offset = 0
                for n in num_masks:
                    results.append(br[offset:offset + n])
                    offset += n
            else:
                results.extend([[]] * input_ids.size(0))
            orders.extend(batch[IDX])
        results = reorder(results, orders)
        if flat:
            results = results[0]
        return results

    def load_config(self, save_dir, filename='config.json', **kwargs):
        self.config.transformer = save_dir

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        self.tokenizer = AutoTokenizer_.from_pretrained(self.config.transformer)

    def load_weights(self, save_dir, filename='model.pt', **kwargs):
        pass
