# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-28 17:33
import datetime
import functools
import logging
import os
from typing import Union, List, Callable

import torch
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from hanlp.common.dataset import SamplerBuilder, SortingSamplerBuilder, PadSequenceDataLoader
from hanlp.common.structure import History
from hanlp.common.torch_component import TorchComponent
from hanlp.common.vocab import Vocab
from hanlp.components.amr.seq2seq.dataset.dataset import AMRDataset, dfs_linearize_tokenize
from hanlp.components.amr.seq2seq.dataset.penman import AMRGraph
from hanlp.components.amr.seq2seq.dataset.tokenization_bart import PENMANBartTokenizer
from hanlp.components.amr.seq2seq.dataset.tokenization_t5 import PENMANT5Tokenizer
from hanlp.components.amr.seq2seq.evaluation import write_predictions, compute_smatch
from hanlp.components.amr.seq2seq.optim import RAdam
from hanlp.layers.transformers.pt_imports import PretrainedConfig, AutoConfig_
from hanlp.layers.transformers.resource import get_model_mirror, get_tokenizer_mirror
from hanlp.metrics.amr.smatch_eval import smatch_eval
from hanlp.metrics.mtl import MetricDict
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.constant import IDX
from hanlp_common.util import merge_locals_kwargs, reorder


class Seq2seq_AMR_Parser(TorchComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._transformer_config: PretrainedConfig = None
        self._tokenizer: PENMANBartTokenizer = None
        self.model: BartForConditionalGeneration = None

    def build_dataloader(self, data, batch_size,
                         gradient_accumulation=1,
                         shuffle=False,
                         sampler_builder: SamplerBuilder = None,
                         device=None,
                         logger: logging.Logger = None,
                         **kwargs) -> DataLoader:
        dataset = self.build_dataset(data, not shuffle)
        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger)
        self.finalize_dataset(dataset, logger)
        if isinstance(data, str):
            dataset.purge_cache()
            timer = CountdownTimer(len(dataset))
            max_num_tokens = 0
            # lc = Counter()
            for each in dataset:
                max_num_tokens = max(max_num_tokens, len(each['text_token_ids']))
                # lc[len(each['text_token_ids'])] += 1
                timer.log(f'Preprocessing and caching samples (longest sequence {max_num_tokens})'
                          f'[blink][yellow]...[/yellow][/blink]')
            # print(lc.most_common())
            if self.vocabs.mutable:
                self.vocabs.lock()
                self.vocabs.summary(logger)

        if not sampler_builder:
            sampler_builder = SortingSamplerBuilder(batch_max_tokens=500)
        sampler = sampler_builder.build([len(x['text_token_ids']) for x in dataset], shuffle,
                                        gradient_accumulation if dataset.cache else 1)
        return self._create_dataloader(dataset, batch_size, device, sampler, shuffle)

    def _create_dataloader(self, dataset, batch_size, device, sampler, shuffle):
        return PadSequenceDataLoader(dataset, batch_size, shuffle, device=device, batch_sampler=sampler,
                                     pad=self._get_pad_dict())

    def _get_pad_dict(self):
        return {'text_token_ids': self._transformer_config.pad_token_id,
                'graph_token_ids': self._transformer_config.pad_token_id}

    def finalize_dataset(self, dataset, logger: logging.Logger = None):
        dataset.append_transform(functools.partial(dfs_linearize_tokenize, tokenizer=self._tokenizer,
                                                   remove_space='chinese' in self.config.transformer))

    def build_dataset(self, data, generate_idx):
        dataset = AMRDataset(data, generate_idx=generate_idx)
        return dataset

    def collect_additional_tokens(self, additional_tokens, dataset):
        pred_min = self.config.pred_min
        frames = dataset.get_frames()
        for token, freq in frames.items():
            if freq >= pred_min:
                additional_tokens.add(token)
        for token, freq in dataset.get_roles().items():
            additional_tokens.add(token)
        additional_tokens.update(self.config.additional_tokens)

    def build_tokenizer(self, additional_tokens) -> PENMANBartTokenizer:
        transformer = self.config.transformer
        if 't5-' in transformer:
            cls = PENMANT5Tokenizer
        elif 'bart-' in transformer:
            cls = PENMANBartTokenizer
        else:
            raise NotImplemented(f'Unsupported transformer {transformer}')
        transformer = get_tokenizer_mirror(transformer)
        self._tokenizer = cls.from_pretrained(
            transformer,
            collapse_name_ops=self.config.collapse_name_ops,
            use_pointer_tokens=self.config.use_pointer_tokens,
            raw_graph=self.config.raw_graph,
            additional_tokens=additional_tokens,
            recategorization_tokens=self.config.recategorization_tokens,
            config=self._transformer_config,
        )
        return self._tokenizer

    def build_optimizer(self, trn, lr, epochs, gradient_accumulation, warmup_steps, weight_decay, **kwargs):
        num_training_steps = len(trn) * epochs // gradient_accumulation
        if isinstance(warmup_steps, float):
            warmup_steps = int(num_training_steps * warmup_steps)
        optimizer = RAdam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay)
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps)
        return optimizer, scheduler

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs):
        pass

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, dev_data=None, eval_after=None,
                              **kwargs):
        best_epoch, best_metric = 0, -1
        if isinstance(eval_after, float):
            eval_after = int(epochs * eval_after)
        timer = CountdownTimer(epochs)
        history = History()
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, history=history, ratio_width=ratio_width,
                                **self.config)
            if epoch > eval_after:
                dev_metric = self.evaluate_dataloader(dev, criterion, logger=logger, ratio_width=ratio_width,
                                                      output=os.path.join(save_dir, 'dev.pred.txt'),
                                                      input=dev_data, use_fast=True)
            timer.update()
            report = f"{timer.elapsed_human} / {timer.total_time_human} ETA: {timer.eta_human}"
            if epoch > eval_after:
                if dev_metric > best_metric:
                    best_epoch, best_metric = epoch, dev_metric
                    self.save_weights(save_dir)
                    report += ' [red](saved)[/red]'
                else:
                    report += f' ({epoch - best_epoch})'
                # if epoch - best_epoch >= patience:
                #     report += ' early stop'
            logger.info(report)
            # if epoch - best_epoch >= patience:
            #     break
        if not best_epoch:
            self.save_weights(save_dir)
        elif best_epoch != epoch:
            self.load_weights(save_dir)
        logger.info(f"Max score of dev is {best_metric} at epoch {best_epoch}")
        logger.info(f"Average time of each epoch is {timer.elapsed_average_human}")
        logger.info(f"{timer.elapsed_human} elapsed")
        return best_metric

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger,
                       history: History = None, gradient_accumulation=1, ratio_percentage=None, **kwargs):
        optimizer, scheduler = optimizer
        self.model.train()
        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation=gradient_accumulation))
        total_loss = 0
        for batch in trn:
            output_dict = self.feed_batch(batch)
            loss = output_dict['loss']
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if history.step(gradient_accumulation):
                self._step(optimizer, scheduler)
                timer.log(self.report_metrics(total_loss / (timer.current + 1)),
                          ratio_percentage=ratio_percentage, logger=logger)
            del loss
            del output_dict
        return total_loss / max(timer.total, 1)

    def _step(self, optimizer, scheduler):
        if self.config.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    def report_metrics(self, loss):
        return f'loss: {loss:.4f}'

    def feed_batch(self, batch):
        input_ids, labels = batch['text_token_ids'], batch.get('graph_token_ids')
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        if labels is not None:
            decoder_input_ids = labels[:, :-1]
            labels = labels[:, 1:].contiguous()
        else:
            decoder_input_ids = None
        return self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                          labels=labels)

    @torch.no_grad()
    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, ratio_width=None,
                            logger=None, input=None, use_fast=False,
                            **kwargs):
        self.model.eval()
        timer = CountdownTimer(len(data))
        graphs = []
        orders = []
        smatch = 0
        for idx, batch in enumerate(data):
            graphs_per_batch = self.predict_amrs(batch)
            graphs_per_batch = [x[0] for x in graphs_per_batch]
            # Copy meta data from gold graph
            for gp, gg in zip(graphs_per_batch, batch['amr']):
                metadata = gg.metadata.copy()
                metadata['annotator'] = f'{self.config.transformer}-amr'
                metadata['date'] = str(datetime.datetime.now())
                if 'save-date' in metadata:
                    del metadata['save-date']
                gp.metadata = metadata
            graphs.extend(graphs_per_batch)
            orders.extend(batch[IDX])
            if idx == timer.total - 1:
                graphs = reorder(graphs, orders)
                write_predictions(output, self._tokenizer, graphs)
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

    def predict_amrs(self, batch, beam_size=1):
        out = self._model_generate(batch, beam_size)
        tokens = []
        for i1 in range(0, out.size(0), beam_size):
            tokens_same_source = []
            tokens.append(tokens_same_source)
            for i2 in range(i1, i1 + beam_size):
                tokk = out[i2].tolist()
                tokens_same_source.append(tokk)
        tokens = [t for tt in tokens for t in tt]
        graphs = []
        tokenizer = self._tokenizer
        for i1 in range(0, len(tokens), beam_size):
            graphs_same_source = []
            graphs.append(graphs_same_source)
            for i2 in range(i1, i1 + beam_size):
                tokk = tokens[i2]
                graph, status, (lin, backr) = tokenizer.decode_amr(tokk, restore_name_ops=False)
                graph.status = status
                graph.nodes = lin
                graph.backreferences = backr
                graph.tokens = tokk
                graphs_same_source.append(graph)
            graphs_same_source[:] = \
                tuple(zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0]))))[1]

        return graphs

    def _model_generate(self, batch, beam_size):
        input_ids = batch['text_token_ids']
        attention_mask = input_ids.ne(self.model.config.pad_token_id).to(torch.long)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            decoder_start_token_id=0,
            num_beams=beam_size,
            num_return_sequences=beam_size)
        return out

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        # noinspection PyTypeChecker
        transformer = self.config.transformer
        cls = self._get_model_cls(transformer)
        transformer = get_model_mirror(self.config.transformer)
        model: cls = cls.from_pretrained(
            transformer,
            config=self._transformer_config) if training else cls(self._transformer_config)
        if not training:
            self.build_tokenizer(self.vocabs['additional_tokens'])
        tokenizer = self._tokenizer
        model.resize_token_embeddings(len(tokenizer.encoder))
        if training:
            self._init_new_embeddings(model if cls == T5ForConditionalGeneration else model.model, tokenizer)
        return model

    def _get_model_cls(self, transformer: str):
        if 't5-' in transformer:
            cls = T5ForConditionalGeneration
        elif 'bart-' in transformer:
            cls = BartForConditionalGeneration
        else:
            raise NotImplemented(f'Unsupported transformer {transformer}')
        return cls

    @staticmethod
    def _init_new_embeddings(model, tokenizer):
        modified = 0
        encoder = tokenizer.encoder
        for tok, idx in encoder.items():
            tok = tok.lstrip(tokenizer.INIT)

            if idx < tokenizer.old_enc_size:
                continue

            elif tok.startswith('<pointer:') and tok.endswith('>'):
                tok_split = ['pointer', str(tok.split(':')[1].strip('>'))]

            elif tok.startswith('<'):
                continue

            elif tok.startswith(':'):
                if tok.startswith(':op'):
                    tok_split = ['relation', 'operator', str(int(tok[3:]))]

                elif tok.startswith(':snt'):
                    tok_split = ['relation', 'sentence', str(int(tok[4:]))]

                elif tok.startswith(':ARG'):
                    tok_split = ['relation', 'argument', str(int(tok[4:]))]
                else:
                    tok_split = ['relation'] + tok.lstrip(':').split('-')
            else:
                tok_split = tok.split('-')

            tok_split_ = tok_split
            tok_split = []
            for s in tok_split_:
                s_ = s + tokenizer.INIT
                if s_ in encoder:
                    tok_split.append(s_)
                else:
                    tok_split.extend(tokenizer._tok_bpe(s))

            vecs = []
            for s in tok_split:
                idx_split = encoder.get(s, -1)
                if idx_split > -1:
                    vec_split = model.encoder.embed_tokens.weight.data[idx_split].clone()
                    vecs.append(vec_split)

            if vecs:
                vec = torch.stack(vecs, 0).mean(0)
                noise = torch.empty_like(vec)
                noise.uniform_(-0.1, +0.1)
                model.encoder.embed_tokens.weight.data[idx] = vec + noise
                modified += 1

    def input_is_flat(self, data):
        return isinstance(data, str)

    def predict(self, data: Union[str, List[str]], beautiful_amr_graph=True, **kwargs):
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        dataloader = self.build_dataloader([{'text': x} for x in data], **self.config, device=self.device)
        orders = []
        results = []
        for batch in dataloader:
            graphs = self.predict_amrs(batch)
            graphs = [x[0] for x in graphs]
            if beautiful_amr_graph:
                graphs = [AMRGraph(x.triples, x.top, x.epidata, x.metadata) for x in graphs]
            results.extend(graphs)
            orders.extend(batch[IDX])
        results = reorder(results, orders)
        if flat:
            results = results[0]
        return results

    def fit(self, trn_data, dev_data, save_dir, batch_size=32, epochs=30,
            transformer='facebook/bart-base',
            lr=5e-05,
            grad_norm=2.5,
            weight_decay=0.004,
            warmup_steps=1,
            dropout=0.25,
            attention_dropout=0.0,
            pred_min=5,
            eval_after=0.5,
            collapse_name_ops=False,
            use_pointer_tokens=True,
            raw_graph=False,
            gradient_accumulation=1,
            recategorization_tokens=(
                    'PERSON', 'COUNTRY', 'QUANTITY', 'ORGANIZATION', 'DATE_ATTRS', 'NATIONALITY', 'LOCATION', 'ENTITY',
                    'CITY',
                    'MISC', 'ORDINAL_ENTITY', 'IDEOLOGY', 'RELIGION', 'STATE_OR_PROVINCE', 'URL', 'CAUSE_OF_DEATH', 'O',
                    'TITLE', 'DATE', 'NUMBER', 'HANDLE', 'SCORE_ENTITY', 'DURATION', 'ORDINAL', 'MONEY', 'SET',
                    'CRIMINAL_CHARGE', '_1', '_2', '_3', '_4', '_2', '_5', '_6', '_7', '_8', '_9', '_10', '_11', '_12',
                    '_13',
                    '_14', '_15'),
            additional_tokens=(
                    'date-entity', 'government-organization', 'temporal-quantity', 'amr-unknown', 'multi-sentence',
                    'political-party', 'monetary-quantity', 'ordinal-entity', 'religious-group', 'percentage-entity',
                    'world-region', 'url-entity', 'political-movement', 'et-cetera', 'at-least', 'mass-quantity',
                    'have-org-role-91', 'have-rel-role-91', 'include-91', 'have-concession-91', 'have-condition-91',
                    'be-located-at-91', 'rate-entity-91', 'instead-of-91', 'hyperlink-91', 'request-confirmation-91',
                    'have-purpose-91', 'be-temporally-at-91', 'regardless-91', 'have-polarity-91', 'byline-91',
                    'have-manner-91', 'have-part-91', 'have-quant-91', 'publication-91', 'be-from-91', 'have-mod-91',
                    'have-frequency-91', 'score-on-scale-91', 'have-li-91', 'be-compared-to-91', 'be-destined-for-91',
                    'course-91', 'have-subevent-91', 'street-address-91', 'have-extent-91', 'statistical-test-91',
                    'have-instrument-91', 'have-name-91', 'be-polite-91', '-00', '-01', '-02', '-03', '-04', '-05',
                    '-06',
                    '-07', '-08', '-09', '-10', '-11', '-12', '-13', '-14', '-15', '-16', '-17', '-18', '-19', '-20',
                    '-21',
                    '-22', '-23', '-24', '-25', '-26', '-27', '-28', '-29', '-20', '-31', '-32', '-33', '-34', '-35',
                    '-36',
                    '-37', '-38', '-39', '-40', '-41', '-42', '-43', '-44', '-45', '-46', '-47', '-48', '-49', '-50',
                    '-51',
                    '-52', '-53', '-54', '-55', '-56', '-57', '-58', '-59', '-60', '-61', '-62', '-63', '-64', '-65',
                    '-66',
                    '-67', '-68', '-69', '-70', '-71', '-72', '-73', '-74', '-75', '-76', '-77', '-78', '-79', '-80',
                    '-81',
                    '-82', '-83', '-84', '-85', '-86', '-87', '-88', '-89', '-90', '-91', '-92', '-93', '-94', '-95',
                    '-96',
                    '-97', '-98', '-of'),
            devices=None,
            logger=None,
            seed=None,
            finetune: Union[bool, str] = False,
            eval_trn=True,
            _device_placeholder=False,
            **kwargs):
        """

        Args:
            trn_data:
            dev_data:
            save_dir:
            batch_size:
            epochs:
            transformer:
            lr:
            grad_norm:
            weight_decay:
            warmup_steps:
            dropout:
            attention_dropout:
            pred_min:
            eval_after:
            collapse_name_ops: ``True`` to merge name ops.
            use_pointer_tokens: ``True`` to use pointer tokens to represent variables.
            raw_graph: ``True`` to use the raw graph as input and skip all pre/post-processing steps.
            gradient_accumulation:
            recategorization_tokens: Tokens used in re-categorization. They will be added to tokenizer too but do not
            put them into ``additional_tokens``.
            additional_tokens: Tokens to be added to the tokenizer vocab.
            devices:
            logger:
            seed:
            finetune:
            eval_trn:
            _device_placeholder:
            **kwargs:

        Returns:

        """
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def on_config_ready(self, **kwargs):
        super().on_config_ready(**kwargs)
        config = AutoConfig_.from_pretrained(self.config.transformer)
        config.output_past = False
        config.no_repeat_ngram_size = 0
        config.prefix = " "
        # config.output_attentions = True
        config.dropout = self.config.dropout
        config.attention_dropout = self.config.attention_dropout
        self._transformer_config = config

    def evaluate(self, tst_data, save_dir=None, logger: logging.Logger = None, batch_size=None, output=True,
                 cache=None, ret_speed=False, **kwargs):
        return super().evaluate(tst_data, save_dir, logger, batch_size, output, cache, ret_speed, **kwargs)

    def build_vocabs(self, trn: torch.utils.data.Dataset, logger: logging.Logger):
        additional_tokens = set()
        self.collect_additional_tokens(additional_tokens, trn)
        additional_tokens = sorted(additional_tokens)
        self.build_tokenizer(additional_tokens)
        self.vocabs['additional_tokens'] = Vocab(idx_to_token=list(additional_tokens))
