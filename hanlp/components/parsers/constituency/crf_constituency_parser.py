# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-28 21:24
import logging
from typing import Union, List

import torch
from phrasetree.tree import Tree
from torch.utils.data import DataLoader

from hanlp_common.constant import BOS, EOS, IDX
from hanlp.common.dataset import TransformableDataset, SamplerBuilder, PadSequenceDataLoader
from hanlp.common.structure import History
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import FieldLength, TransformList
from hanlp.common.vocab import VocabWithNone
from hanlp.components.classifiers.transformer_classifier import TransformerComponent
from hanlp.components.parsers.constituency.constituency_dataset import ConstituencyDataset, unpack_tree_to_features, \
    build_tree, factorize, remove_subcategory
from hanlp.components.parsers.constituency.crf_constituency_model import CRFConstituencyDecoder, CRFConstituencyModel
from hanlp.metrics.parsing.span import SpanMetric
from hanlp.utils.time_util import CountdownTimer
from hanlp.utils.torch_util import clip_grad_norm
from hanlp_common.util import merge_locals_kwargs, merge_dict, reorder


class CRFConstituencyParser(TorchComponent):
    def __init__(self, **kwargs) -> None:
        """Two-stage CRF Parsing (:cite:`ijcai2020-560`).

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)
        self.model: CRFConstituencyModel = self.model

    def build_optimizer(self, trn, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        return TransformerComponent.build_optimizer(self, trn, **kwargs)

    def build_criterion(self, decoder=None, **kwargs):
        return decoder

    def build_metric(self, **kwargs):
        return SpanMetric()

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, patience=0.5, eval_trn=True, **kwargs):
        if isinstance(patience, float):
            patience = int(patience * epochs)
        best_epoch, best_metric = 0, -1
        timer = CountdownTimer(epochs)
        history = History()
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, history=history, ratio_width=ratio_width,
                                eval_trn=eval_trn, **self.config)
            loss, dev_metric = self.evaluate_dataloader(dev, criterion, logger=logger, ratio_width=ratio_width)
            timer.update()
            report = f"{timer.elapsed_human} / {timer.total_time_human} ETA: {timer.eta_human}"
            if dev_metric > best_metric:
                best_epoch, best_metric = epoch, dev_metric
                self.save_weights(save_dir)
                report += ' [red](saved)[/red]'
            else:
                report += f' ({epoch - best_epoch})'
                if epoch - best_epoch >= patience:
                    report += ' early stop'
            logger.info(report)
            if epoch - best_epoch >= patience:
                break
        if not best_epoch:
            self.save_weights(save_dir)
        elif best_epoch != epoch:
            self.load_weights(save_dir)
        logger.info(f"Max score of dev is {best_metric} at epoch {best_epoch}")
        logger.info(f"Average time of each epoch is {timer.elapsed_average_human}")
        logger.info(f"{timer.elapsed_human} elapsed")

    # noinspection PyMethodOverriding
    def fit_dataloader(self,
                       trn: DataLoader,
                       criterion,
                       optimizer,
                       metric: SpanMetric,
                       logger: logging.Logger,
                       history: History,
                       gradient_accumulation=1,
                       grad_norm=None,
                       ratio_width=None,
                       eval_trn=True,
                       **kwargs):
        optimizer, scheduler = optimizer
        metric.reset()
        self.model.train()
        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation=gradient_accumulation))
        total_loss = 0
        for idx, batch in enumerate(trn):
            out, mask = self.feed_batch(batch)
            y = batch['chart_id']
            loss, span_probs = self.compute_loss(out, y, mask)
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if eval_trn:
                prediction = self.decode_output(out, mask, batch, span_probs)
                self.update_metrics(metric, batch, prediction)
            if history.step(gradient_accumulation):
                self._step(optimizer, scheduler, grad_norm)
                report = f'loss: {total_loss / (idx + 1):.4f} {metric}' if eval_trn \
                    else f'loss: {total_loss / (idx + 1):.4f}'
                timer.log(report, logger=logger, ratio_percentage=False, ratio_width=ratio_width)
            del loss
            del out
            del mask

    def decode_output(self, out, mask, batch, span_probs=None, decoder=None, tokens=None):
        s_span, s_label = out
        if not decoder:
            decoder = self.model.decoder
        if span_probs is None:
            if self.config.mbr:
                s_span = decoder.crf(s_span, mask, mbr=True)
        else:
            s_span = span_probs
        chart_preds = decoder.decode(s_span, s_label, mask)
        idx_to_token = self.vocabs.chart.idx_to_token
        if tokens is None:
            tokens = [x[1:-1] for x in batch['token']]
        trees = [build_tree(token, [(i, j, idx_to_token[label]) for i, j, label in chart]) for token, chart in
                 zip(tokens, chart_preds)]
        # probs = [prob[:i - 1, 1:i].cpu() for i, prob in zip(lens, s_span.unbind())]
        return trees

    def update_metrics(self, metric, batch, prediction):
        # Add pre-terminals (pos tags) back to prediction for safe factorization (deletion based on pos)
        for pred, gold in zip(prediction, batch['constituency']):
            pred: Tree = pred
            gold: Tree = gold
            for p, g in zip(pred.subtrees(lambda t: t.height() == 2), gold.pos()):
                token, pos = g
                p: Tree = p
                assert p.label() == '_'
                p.set_label(pos)
        metric([factorize(tree, self.config.delete, self.config.equal) for tree in prediction],
               [factorize(tree, self.config.delete, self.config.equal) for tree in batch['constituency']])
        return metric

    def feed_batch(self, batch: dict):
        mask = self.compute_mask(batch)
        s_span, s_label = self.model(batch)
        return (s_span, s_label), mask

    def compute_mask(self, batch, offset=1):
        lens = batch['token_length'] - offset
        seq_len = lens.max()
        mask = lens.new_tensor(range(seq_len)) < lens.view(-1, 1, 1)
        mask = mask & mask.new_ones(seq_len, seq_len).triu_(1)
        return mask

    def compute_loss(self, out, y, mask, crf_decoder=None):
        if not crf_decoder:
            crf_decoder = self.model.decoder
        loss, span_probs = crf_decoder.loss(out[0], out[1], y, mask, self.config.mbr)
        if loss < 0:  # wired negative loss
            loss *= 0
        return loss, span_probs

    def _step(self, optimizer, scheduler, grad_norm):
        clip_grad_norm(self.model, grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    @torch.no_grad()
    def evaluate_dataloader(self, data, criterion, logger=None, ratio_width=None, metric=None, output=None, **kwargs):
        self.model.eval()
        total_loss = 0
        if not metric:
            metric = self.build_metric()
        else:
            metric.reset()
        timer = CountdownTimer(len(data))
        for idx, batch in enumerate(data):
            out, mask = self.feed_batch(batch)
            y = batch['chart_id']
            loss, span_probs = self.compute_loss(out, y, mask)
            total_loss += loss.item()
            prediction = self.decode_output(out, mask, batch, span_probs)
            self.update_metrics(metric, batch, prediction)
            timer.log(f'loss: {total_loss / (idx + 1):.4f} {metric}', ratio_percentage=False, logger=logger,
                      ratio_width=ratio_width)
        total_loss /= len(data)
        if output:
            output.close()
        return total_loss, metric

    # noinspection PyMethodOverriding
    def build_model(self, encoder, training=True, **kwargs) -> torch.nn.Module:
        decoder = CRFConstituencyDecoder(n_labels=len(self.vocabs.chart), n_hidden=encoder.get_output_dim(), **kwargs)
        encoder = encoder.module(vocabs=self.vocabs, training=training)
        return CRFConstituencyModel(encoder, decoder)

    def build_dataloader(self,
                         data,
                         batch_size,
                         sampler_builder: SamplerBuilder = None,
                         gradient_accumulation=1,
                         shuffle=False,
                         device=None,
                         logger: logging.Logger = None,
                         **kwargs) -> DataLoader:
        if isinstance(data, TransformableDataset):
            dataset = data
        else:
            transform = self.config.encoder.transform()
            if self.config.get('transform', None):
                transform = TransformList(self.config.transform, transform)
            dataset = self.build_dataset(data, transform, logger)
        if self.vocabs.mutable:
            # noinspection PyTypeChecker
            self.build_vocabs(dataset, logger)
        lens = [len(x['token_input_ids']) for x in dataset]
        if sampler_builder:
            sampler = sampler_builder.build(lens, shuffle, gradient_accumulation)
        else:
            sampler = None
        return PadSequenceDataLoader(dataset, batch_size, shuffle, device=device, batch_sampler=sampler)

    def predict(self, data: Union[str, List[str]], batch_size: int = None, **kwargs):
        if not data:
            return []
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        samples = self.build_samples(data)
        dataloader = self.build_dataloader(samples, device=self.device,
                                           **merge_dict(self.config, batch_size=batch_size, overwrite=True))
        outputs = []
        orders = []
        for idx, batch in enumerate(dataloader):
            out, mask = self.feed_batch(batch)
            prediction = self.decode_output(out, mask, batch, span_probs=None)
            # prediction = [x[0] for x in prediction]
            outputs.extend(prediction)
            orders.extend(batch[IDX])
        outputs = reorder(outputs, orders)
        if flat:
            return outputs[0]
        return outputs

    def input_is_flat(self, data):
        return isinstance(data[0], str)

    def build_samples(self, data):
        return [{'token': [BOS] + token + [EOS]} for token in data]

    # noinspection PyMethodOverriding
    def fit(self,
            trn_data,
            dev_data,
            save_dir,
            encoder,
            lr=5e-5,
            transformer_lr=None,
            adam_epsilon=1e-8,
            weight_decay=0,
            warmup_steps=0.1,
            grad_norm=1.0,
            n_mlp_span=500,
            n_mlp_label=100,
            mlp_dropout=.33,
            batch_size=None,
            batch_max_tokens=5000,
            gradient_accumulation=1,
            epochs=30,
            patience=0.5,
            mbr=True,
            sampler_builder=None,
            delete=('', ':', '``', "''", '.', '?', '!', '-NONE-', 'TOP', ',', 'S1'),
            equal=(('ADVP', 'PRT'),),
            no_subcategory=True,
            eval_trn=True,
            transform=None,
            devices=None,
            logger=None,
            seed=None,
            **kwargs):
        if isinstance(equal, tuple):
            equal = dict(equal)
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_dataset(self, data, transform, logger=None):
        _transform = [
            unpack_tree_to_features,
            self.vocabs,
            FieldLength('token'),
            transform
        ]
        if self.config.get('no_subcategory', True):
            _transform.insert(0, remove_subcategory)
        dataset = ConstituencyDataset(data,
                                      transform=_transform,
                                      cache=isinstance(data, str))
        return dataset

    def build_vocabs(self, trn, logger, **kwargs):
        self.vocabs.chart = VocabWithNone(pad_token=None, unk_token=None)
        timer = CountdownTimer(len(trn))
        max_seq_len = 0
        for each in trn:
            max_seq_len = max(max_seq_len, len(each['token_input_ids']))
            timer.log(f'Building vocab [blink][yellow]...[/yellow][/blink] (longest sequence: {max_seq_len})')
        self.vocabs.chart.set_unk_as_safe_unk()
        self.vocabs.lock()
        self.vocabs.summary(logger)
