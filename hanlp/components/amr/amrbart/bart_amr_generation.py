# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-12-05 17:56
import logging
import os.path
from typing import Callable, Union, List

import penman
import torch
from torch.utils.data import DataLoader

from hanlp.components.amr.amrbart.data_interface.dataset import AMR2TextDataSet
from hanlp.common.dataset import SortingSamplerBuilder, PadSequenceDataLoader
from hanlp.common.torch_component import TorchComponent
from hanlp.components.amr.seq2seq.dataset.dataset import AMRDataset
from hanlp.layers.transformers.pt_imports import AutoConfig_
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.constant import IDX
from hanlp_common.util import reorder
from hanlp.components.amr.amrbart.model_interface.modeling_bart import BartForConditionalGeneration
from hanlp.components.amr.amrbart.model_interface.tokenization_bart import AMRBartTokenizer
from hanlp.components.amr.amrbart.preprocess.read_and_process import dfs_linearize


class BART_AMR_Generation(TorchComponent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer: AMRBartTokenizer = None
        self.transformer_config = None
        self.model: BartForConditionalGeneration = None

    def build_dataloader(self, data, batch_size=32, shuffle=False, device=None, logger: logging.Logger = None,
                         sampler_builder=None,
                         **kwargs) -> DataLoader:
        dataset = AMRDataset(data, generate_idx=True, cache=True)
        dataset.append_transform(lambda x: {**x, 'lamr': ' '.join(dfs_linearize(x['amr']))})
        dataset.append_transform(
            lambda x: AMR2TextDataSet.tokenize(x, tokenizer=self.tokenizer, text='text', amr='lamr')
        )
        if not sampler_builder:
            sampler_builder = SortingSamplerBuilder(batch_max_tokens=500)
        sampler = sampler_builder.build([len(x['input_ids']) for x in dataset], shuffle, 1)
        return PadSequenceDataLoader(dataset, batch_size, shuffle, device=device, batch_sampler=sampler,
                                     pad={'input_ids': self.transformer_config.pad_token_id,
                                          'labels': self.transformer_config.pad_token_id})

    def build_optimizer(self, **kwargs):
        pass

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs):
        pass

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, **kwargs):
        pass

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        pass

    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, **kwargs):
        pass

    def build_model(self, training=True, transformer=None, **kwargs) -> torch.nn.Module:
        model = BartForConditionalGeneration.from_pretrained(
            transformer,
            config=self.transformer_config,
        )
        if not training:
            model.eval()
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def input_is_flat(self, data):
        return isinstance(data, (str, penman.Graph))

    def predict(
            self,
            data: Union[str, List[str]], num_beams=5, max_length=1024, beautiful_amr_graph=True, verbose=False,
            **kwargs
    ):
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        dataloader = self.build_dataloader([{'amr': penman.loads(x)[0] if isinstance(x, str) else x} for x in data],
                                           **self.config, device=self.device)
        orders = []
        results = []
        if verbose:
            timer = CountdownTimer(len(dataloader))
        for batch in dataloader:
            pieces = self.predict_batch(batch, num_beams, max_length)
            results.extend(pieces)
            orders.extend(batch[IDX])
            if verbose:
                # noinspection PyUnboundLocalVariable
                timer.log()
        results = reorder(results, orders)
        if flat:
            results = results[0]
        return results

    def predict_batch(self, batch, num_beams, max_length):
        tokenizer = self.tokenizer
        input_ids = batch['input_ids']
        preds = self.model.generate(
            input_ids,
            num_beams=num_beams,
            use_cache=True,
            decoder_start_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=0,
            max_length=max_length,
            min_length=0,
            length_penalty=1.0,
        )
        # tokens = batch['tgt']
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [x.strip() for x in decoded_preds]
        return decoded_preds

    def load_config(self, save_dir: str, filename='config.json', **kwargs):
        if os.path.isdir(save_dir):
            super().load_config(save_dir, filename, **kwargs)
            transformer = self.config.transformer
        else:
            self.config.transformer = transformer = save_dir
        self.transformer_config = AutoConfig_.from_pretrained(transformer)

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        self.tokenizer = AMRBartTokenizer.from_pretrained(
            self.config.transformer,
            use_fast=True,
        )

    def load_weights(self, save_dir, filename='model.pt', **kwargs):
        pass
