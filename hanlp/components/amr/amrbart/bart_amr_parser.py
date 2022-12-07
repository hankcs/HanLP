# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-12-05 17:56
import logging
import os.path
from typing import Callable, Union, List

import datetime
import torch
from torch.utils.data import DataLoader

from hanlp.components.amr.amrbart.data_interface.dataset import AMRParsingDataSet
from hanlp.common.dataset import SortingSamplerBuilder, PadSequenceDataLoader
from hanlp.common.torch_component import TorchComponent
from hanlp.components.amr.seq2seq.dataset.dataset import AMRDataset
from hanlp.components.amr.seq2seq.dataset.penman import AMRGraph
from hanlp.components.amr.seq2seq.evaluation import write_predictions, compute_smatch
from hanlp.layers.transformers.pt_imports import AutoConfig_
from hanlp.metrics.amr.smatch_eval import smatch_eval
from hanlp.metrics.mtl import MetricDict
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.constant import IDX
from hanlp_common.util import reorder
from hanlp.components.amr.amrbart.model_interface.modeling_bart import BartForConditionalGeneration
from hanlp.components.amr.amrbart.model_interface.tokenization_bart import AMRBartTokenizer


class BART_AMR_Parser(TorchComponent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer: AMRBartTokenizer = None
        self.transformer_config = None
        self.model: BartForConditionalGeneration = None

    def build_dataloader(self, data, batch_size=32, shuffle=False, device=None, logger: logging.Logger = None,
                         sampler_builder=None,
                         **kwargs) -> DataLoader:
        dataset = AMRDataset(data, generate_idx=True, cache=True)
        if isinstance(data, str):
            dataset.append_transform(lambda x: {**x, 'text': x['amr'].metadata['snt']})
        dataset.append_transform(
            lambda x: AMRParsingDataSet.tokenize(x, tokenizer=self.tokenizer, text='text')
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
        return isinstance(data, str)

    def predict(
            self,
            data: Union[str, List[str]], num_beams=5, max_length=1024, beautiful_amr_graph=True, verbose=False,
            **kwargs
    ):
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        dataloader = self.build_dataloader([{'text': x} for x in data], **self.config, device=self.device)
        orders = []
        results = []
        # inputs, logits, labels, loss = torch.load('/local/scratch/hhe43/amrbart/batch.pt')
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
            num_return_sequences=num_beams,
            use_cache=True,
            decoder_start_token_id=tokenizer.amr_bos_token_id,
            eos_token_id=tokenizer.amr_eos_token_id,
            no_repeat_ngram_size=0,
            max_length=max_length,
            min_length=0,
            length_penalty=1.0,
        ).tolist()
        # tokens = batch['tgt']
        graphs = []
        for i in range(0, len(preds), num_beams):
            graphs_same_source = []
            for j in range(i, i + num_beams):
                ith_pred = preds[j]
                ith_pred[0] = tokenizer.bos_token_id
                ith_pred = [
                    tokenizer.eos_token_id if itm == tokenizer.amr_eos_token_id else itm
                    for itm in ith_pred if itm != tokenizer.pad_token_id
                ]

                graph, status, (lin, backr) = tokenizer.decode_amr(
                    ith_pred, restore_name_ops=False
                )
                graph.status = status
                graph.nodes = lin
                graph.backreferences = backr
                graph.tokens = ith_pred
                graphs_same_source.append(graph)
            graphs_same_source[:] = \
                tuple(zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0]))))[1]
            graphs.append(graphs_same_source)
        # assert len(graphs) == len(tokens), f"inconsistent lengths {len(graphs)} vs {len(tokens)}"
        # for idx, gps, snt in zip(batch[IDX], graphs, tokens):
        #     for gp in gps:
        #         gp.metadata = {"id": str(idx), "annotator": "bart-amr",
        #                        "snt": snt.replace("<AMR>", '').replace("</AMR>", '').strip()}
        pieces = [AMRGraph(g.triples, g.top, g.epidata, g.metadata) for g in [gs[0] for gs in graphs]]
        return pieces

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

    @torch.no_grad()
    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, ratio_width=None,
                            logger=None, input=None, use_fast=False, num_beams=5, max_length=1024,
                            **kwargs):
        self.model.eval()
        timer = CountdownTimer(len(data))
        graphs = []
        orders = []
        smatch = 0
        for idx, batch in enumerate(data):
            graphs_per_batch = self.predict_batch(batch, num_beams, max_length)
            # Copy meta data from gold graph
            for gp, gg in zip(graphs_per_batch, batch['amr']):
                metadata = gg.metadata.copy()
                metadata['annotator'] = f'{self.transformer_config.name_or_path}-amr'
                metadata['date'] = str(datetime.datetime.now())
                if 'save-date' in metadata:
                    del metadata['save-date']
                gp.metadata = metadata
            graphs.extend(graphs_per_batch)
            orders.extend(batch[IDX])
            if idx == timer.total - 1:
                graphs = reorder(graphs, orders)
                write_predictions(output, None, graphs)
                try:
                    if use_fast:
                        smatch = compute_smatch(output, input)
                    else:
                        smatch = smatch_eval(output, input, use_fast=False)
                except:
                    pass
                timer.log(smatch.cstr() if isinstance(smatch, MetricDict) else f'{smatch:.2%}', ratio_percentage=False,
                          logger=logger)
            else:
                timer.log(ratio_percentage=False, logger=logger)

        return smatch

    def evaluate(self, tst_data, save_dir=None, logger: logging.Logger = None, batch_size=None, output=True, **kwargs):
        return super().evaluate(tst_data, save_dir, logger, batch_size, output, **kwargs)
