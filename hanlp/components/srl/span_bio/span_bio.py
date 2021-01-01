# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-22 20:54
import logging
from copy import copy
from typing import Union, List, Callable, Dict, Any
from bisect import bisect
import torch
import torch.nn.functional as F
from alnlp.modules.util import lengths_to_mask
from torch import nn
from torch.utils.data import DataLoader

from hanlp_common.constant import IDX, PRED
from hanlp.common.dataset import PadSequenceDataLoader, SamplerBuilder, TransformableDataset
from hanlp.common.structure import History
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import FieldLength
from hanlp.common.vocab import Vocab
from hanlp.components.srl.span_bio.baffine_tagging import SpanBIOSemanticRoleLabelingModel
from hanlp.datasets.srl.conll2012 import CoNLL2012SRLBIODataset
from hanlp.layers.crf.crf import CRF
from hanlp.layers.embeddings.contextual_word_embedding import find_transformer
from hanlp.layers.embeddings.embedding import Embedding
from hanlp.layers.transformers.utils import build_optimizer_scheduler_with_transformer
from hanlp.metrics.chunking.sequence_labeling import get_entities
from hanlp.metrics.f1 import F1
from hanlp.utils.string_util import guess_delimiter
from hanlp.utils.time_util import CountdownTimer
from hanlp.utils.torch_util import clip_grad_norm
from hanlp_common.util import merge_locals_kwargs, reorder


class SpanBIOSemanticRoleLabeler(TorchComponent):

    def __init__(self, **kwargs) -> None:
        """A span based Semantic Role Labeling task using BIO scheme for tagging the role of each token. Given a
        predicate and a token, it uses biaffine (:cite:`dozat:17a`) to predict their relations as one of BIO-ROLE.

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)
        self.model: SpanBIOSemanticRoleLabelingModel = None

    def build_optimizer(self,
                        trn,
                        epochs,
                        lr,
                        adam_epsilon,
                        weight_decay,
                        warmup_steps,
                        transformer_lr=None,
                        gradient_accumulation=1,
                        **kwargs):
        num_training_steps = len(trn) * epochs // gradient_accumulation
        if transformer_lr is None:
            transformer_lr = lr
        transformer = find_transformer(self.model.embed)
        optimizer, scheduler = build_optimizer_scheduler_with_transformer(self.model, transformer,
                                                                          lr, transformer_lr,
                                                                          num_training_steps, warmup_steps,
                                                                          weight_decay, adam_epsilon)
        return optimizer, scheduler

    def build_criterion(self, decoder=None, **kwargs):
        if self.config.crf:
            if not decoder:
                decoder = self.model.decoder
            if isinstance(decoder, torch.nn.DataParallel):
                decoder = decoder.module
            return decoder.crf
        else:
            return nn.CrossEntropyLoss(reduction=self.config.loss_reduction)

    def build_metric(self, **kwargs):
        return F1()

    def execute_training_loop(self,
                              trn: DataLoader,
                              dev: DataLoader,
                              epochs,
                              criterion,
                              optimizer,
                              metric,
                              save_dir,
                              logger: logging.Logger,
                              devices,
                              ratio_width=None,
                              patience=0.5,
                              **kwargs):
        if isinstance(patience, float):
            patience = int(patience * epochs)
        best_epoch, best_metric = 0, -1
        timer = CountdownTimer(epochs)
        history = History()
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, history=history, ratio_width=ratio_width,
                                **self.config)
            loss, dev_metric = self.evaluate_dataloader(dev, criterion, metric, logger=logger, ratio_width=ratio_width)
            timer.update()
            report = f"{timer.elapsed_human} / {timer.total_time_human} ETA: {timer.eta_human}"
            if dev_metric > best_metric:
                best_epoch, best_metric = epoch, copy(dev_metric)
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
                       metric,
                       logger: logging.Logger,
                       history: History,
                       gradient_accumulation=1,
                       grad_norm=None,
                       ratio_width=None,
                       eval_trn=False,
                       **kwargs):
        optimizer, scheduler = optimizer
        self.model.train()
        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation=gradient_accumulation))
        total_loss = 0
        for idx, batch in enumerate(trn):
            pred, mask = self.feed_batch(batch)
            loss = self.compute_loss(criterion, pred, batch['srl_id'], mask)
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if eval_trn:
                prediction = self.decode_output(pred, mask, batch)
                self.update_metrics(metric, prediction, batch)
            if history.step(gradient_accumulation):
                self._step(optimizer, scheduler, grad_norm)
                report = f'loss: {total_loss / (idx + 1):.4f} {metric}' if eval_trn else f'loss: {total_loss / (idx + 1):.4f}'
                timer.log(report, logger=logger, ratio_percentage=False, ratio_width=ratio_width)
            del loss
            del pred
            del mask

    def naive_decode(self, pred, mask, batch, decoder=None):
        vocab = self.vocabs['srl'].idx_to_token
        results = []
        for sent, matrix in zip(batch['token'], pred.argmax(-1).tolist()):
            results.append([])
            for token, tags_per_token in zip(sent, matrix):
                tags_per_token = [vocab[x] for x in tags_per_token][:len(sent)]
                srl_per_token = get_entities(tags_per_token)
                results[-1].append(srl_per_token)
        return results

    def decode_output(self, pred, mask, batch, decoder=None):
        # naive = self.naive_decode(pred, mask, batch, decoder)
        vocab = self.vocabs['srl'].idx_to_token
        if self.config.crf:
            if not decoder:
                decoder = self.model.decoder
            crf: CRF = decoder.crf
            token_index, mask = mask
            pred = crf.decode(pred, mask)
            pred = sum(pred, [])
        else:
            pred = pred[mask].argmax(-1)
            pred = pred.tolist()
        pred = [vocab[x] for x in pred]
        results = []
        offset = 0
        for sent in batch['token']:
            results.append([])
            for token in sent:
                tags_per_token = pred[offset:offset + len(sent)]
                srl_per_token = get_entities(tags_per_token)
                results[-1].append(srl_per_token)
                offset += len(sent)
        assert offset == len(pred)
        # assert results == naive
        return results

    def update_metrics(self, metric, prediction, batch):
        for p, g in zip(prediction, batch['srl_set']):
            srl = set()
            for i, args in enumerate(p):
                srl.update((i, start, end, label) for (label, start, end) in args)
            metric(srl, g)
        return metric

    def feed_batch(self, batch: dict):
        lens = batch['token_length']
        mask2d = lengths_to_mask(lens)
        pred = self.model(batch, mask=mask2d)
        mask3d = self.compute_mask(mask2d)
        if self.config.crf:
            token_index = mask3d[0]
            pred = pred.flatten(end_dim=1)[token_index]
            pred = F.log_softmax(pred, dim=-1)
        return pred, mask3d

    def compute_mask(self, mask2d):
        mask3d = mask2d.unsqueeze_(-1).expand(-1, -1, mask2d.size(1))
        mask3d = mask3d & mask3d.transpose(1, 2)
        if self.config.crf:
            mask3d = mask3d.flatten(end_dim=1)
            token_index = mask3d[:, 0]
            mask3d = mask3d[token_index]
            return token_index, mask3d
        else:
            return mask3d

    def _step(self, optimizer, scheduler, grad_norm):
        clip_grad_norm(self.model, grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # noinspection PyMethodOverriding
    def build_model(self, embed: Embedding, encoder, training, **kwargs) -> torch.nn.Module:
        # noinspection PyCallByClass
        model = SpanBIOSemanticRoleLabelingModel(
            embed.module(training=training, vocabs=self.vocabs),
            encoder,
            len(self.vocabs.srl),
            self.config.n_mlp_rel,
            self.config.mlp_dropout,
            self.config.crf,
        )
        return model

    # noinspection PyMethodOverriding
    def build_dataloader(self, data, batch_size,
                         sampler_builder: SamplerBuilder = None,
                         gradient_accumulation=1,
                         shuffle=False, device=None, logger: logging.Logger = None,
                         **kwargs) -> DataLoader:
        if isinstance(data, TransformableDataset):
            dataset = data
        else:
            dataset = self.build_dataset(data, [self.config.embed.transform(vocabs=self.vocabs), self.vocabs,
                                                FieldLength('token')])
        if self.vocabs.mutable:
            # noinspection PyTypeChecker
            self.build_vocabs(dataset, logger)
        lens = [len(x['token_input_ids']) for x in dataset]
        if sampler_builder:
            sampler = sampler_builder.build(lens, shuffle, gradient_accumulation)
        else:
            sampler = None
        return PadSequenceDataLoader(dataset, batch_size, shuffle, device=device, batch_sampler=sampler)

    def build_dataset(self, data, transform):
        dataset = CoNLL2012SRLBIODataset(data,
                                         transform=transform,
                                         doc_level_offset=self.config.get('doc_level_offset', True),
                                         cache=isinstance(data, str))
        return dataset

    def build_vocabs(self, dataset, logger, **kwargs):
        self.vocabs.srl = Vocab(pad_token=None, unk_token=None)
        timer = CountdownTimer(len(dataset))
        max_seq_len = 0
        for sample in dataset:
            max_seq_len = max(max_seq_len, len(sample['token_input_ids']))
            timer.log(f'Building vocab [blink][yellow]...[/yellow][/blink] (longest sequence: {max_seq_len})')
        self.vocabs['srl'].set_unk_as_safe_unk()  # C-ARGM-FRQ appears only in test set
        self.vocabs.lock()
        self.vocabs.summary(logger)
        if self.config.get('delimiter') is None:
            tokens = dataset[0]['token']
            self.config.delimiter = guess_delimiter(tokens)
            logger.info(f'Guess the delimiter between tokens could be [blue]"{self.config.delimiter}"[/blue]. '
                        f'If not, specify `delimiter` in `fit()`')

    def predict(self, data: Union[str, List[str]], batch_size: int = None, **kwargs):
        if not data:
            return []
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        dataloader = self.build_dataloader(self.build_samples(data), batch_size, device=self.device, **kwargs)
        results = []
        order = []
        for batch in dataloader:
            pred, mask = self.feed_batch(batch)
            prediction = self.decode_output(pred, mask, batch)
            results.extend(self.prediction_to_result(prediction, batch))
            order.extend(batch[IDX])
        results = reorder(results, order)
        if flat:
            return results[0]
        return results

    def build_samples(self, data):
        return [{'token': token} for token in data]

    # noinspection PyMethodOverriding
    def fit(self,
            trn_data,
            dev_data,
            save_dir,
            embed,
            encoder=None,
            lr=1e-3,
            transformer_lr=1e-4,
            adam_epsilon=1e-8,
            warmup_steps=0.1,
            weight_decay=0,
            crf=False,
            n_mlp_rel=300,
            mlp_dropout=0.2,
            batch_size=32,
            gradient_accumulation=1,
            grad_norm=1,
            loss_reduction='mean',
            epochs=30,
            delimiter=None,
            doc_level_offset=True,
            eval_trn=False,
            logger=None,
            devices: Union[float, int, List[int]] = None,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def compute_loss(self, criterion, pred, srl, mask):
        if self.config.crf:
            token_index, mask = mask
            criterion: CRF = criterion
            loss = -criterion.forward(pred, srl.flatten(end_dim=1)[token_index], mask,
                                      reduction=self.config.loss_reduction)
        else:
            loss = criterion(pred[mask], srl[mask])
        return loss

    # noinspection PyMethodOverriding
    @torch.no_grad()
    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric, logger, ratio_width=None,
                            filename=None, **kwargs):
        self.model.eval()
        timer = CountdownTimer(len(data))
        total_loss = 0
        metric.reset()
        for idx, batch in enumerate(data):
            pred, mask = self.feed_batch(batch)
            loss = self.compute_loss(criterion, pred, batch['srl_id'], mask)
            total_loss += loss.item()
            prediction = self.decode_output(pred, mask, batch)
            self.update_metrics(metric, prediction, batch)
            report = f'loss: {total_loss / (idx + 1):.4f} {metric}'
            timer.log(report, logger=logger, ratio_percentage=False, ratio_width=ratio_width)
        return total_loss / timer.total, metric

    def input_is_flat(self, data) -> bool:
        return isinstance(data[0], str)

    def prediction_to_result(self, prediction: List, batch: Dict[str, Any], delimiter=None) -> List:
        if delimiter is None:
            delimiter = self.config.delimiter
        for matrix, tokens in zip(prediction, batch['token']):
            result = []
            for i, arguments in enumerate(matrix):
                if arguments:
                    pas = [(delimiter.join(tokens[x[1]:x[2]]),) + x for x in arguments]
                    pas.insert(bisect([a[1] for a in arguments], i), (tokens[i], PRED, i, i + 1))
                    result.append(pas)
            yield result
