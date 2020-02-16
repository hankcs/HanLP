# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-14 20:34
import logging
from copy import deepcopy
from typing import Union, List, Callable

import torch
from alnlp.modules.util import lengths_to_mask
from torch.utils.data import DataLoader

from hanlp_common.constant import IDX
from hanlp.common.dataset import PadSequenceDataLoader, SortingSamplerBuilder
from hanlp.common.structure import History
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import FieldLength, PunctuationMask
from hanlp.common.vocab import Vocab
from hanlp.components.classifiers.transformer_classifier import TransformerComponent
from hanlp.components.parsers.biaffine.biaffine_dep import BiaffineDependencyParser
from hanlp_common.conll import CoNLLUWord, CoNLLSentence
from hanlp.components.parsers.ud.ud_model import UniversalDependenciesModel
from hanlp.components.parsers.ud.util import generate_lemma_rule, append_bos, sample_form_missing
from hanlp.components.parsers.ud.lemma_edit import apply_lemma_rule
from hanlp.datasets.parsing.conll_dataset import CoNLLParsingDataset
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding
from hanlp.metrics.accuracy import CategoricalAccuracy
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp.metrics.parsing.attachmentscore import AttachmentScore
from hanlp.utils.time_util import CountdownTimer
from hanlp.utils.torch_util import clip_grad_norm
from hanlp_common.util import merge_locals_kwargs, merge_dict, reorder


class UniversalDependenciesParser(TorchComponent):

    def __init__(self, **kwargs) -> None:
        """Universal Dependencies Parsing (lemmatization, features, PoS tagging and dependency parsing) implementation
        of "75 Languages, 1 Model: Parsing Universal Dependencies Universally" (:cite:`kondratyuk-straka-2019-75`).

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)
        self.model: UniversalDependenciesModel = self.model

    def build_dataloader(self,
                         data,
                         batch_size,
                         shuffle=False,
                         device=None,
                         logger: logging.Logger = None,
                         sampler_builder=None,
                         gradient_accumulation=1,
                         transformer: ContextualWordEmbedding = None,
                         **kwargs) -> DataLoader:
        transform = [generate_lemma_rule, append_bos, self.vocabs, transformer.transform(), FieldLength('token')]
        if not self.config.punct:
            transform.append(PunctuationMask('token', 'punct_mask'))
        dataset = self.build_dataset(data, transform)
        if self.vocabs.mutable:
            # noinspection PyTypeChecker
            self.build_vocabs(dataset, logger)
        lens = [len(x['token_input_ids']) for x in dataset]
        if sampler_builder:
            sampler = sampler_builder.build(lens, shuffle, gradient_accumulation)
        else:
            sampler = SortingSamplerBuilder(batch_size).build(lens, shuffle, gradient_accumulation)
        return PadSequenceDataLoader(dataset, batch_size, shuffle, device=device, batch_sampler=sampler,
                                     pad={'arc': 0}, )

    def build_vocabs(self, trn, logger, **kwargs):
        self.vocabs.pos = Vocab(unk_token=None, pad_token=None)
        self.vocabs.rel = Vocab(unk_token=None, pad_token=None)
        self.vocabs.lemma = Vocab(unk_token=None, pad_token=None)
        self.vocabs.feat = Vocab(unk_token=None, pad_token=None)
        timer = CountdownTimer(len(trn))
        max_seq_len = 0
        for each in trn:
            max_seq_len = max(max_seq_len, len(each['token']))
            timer.log(f'Building vocab [blink][yellow]...[/yellow][/blink] (longest sequence: {max_seq_len})')
        for v in self.vocabs.values():
            v.set_unk_as_safe_unk()
        self.vocabs.lock()
        self.vocabs.summary(logger)

    def build_dataset(self, data, transform):
        dataset = CoNLLParsingDataset(data, transform=transform, prune=sample_form_missing, cache=isinstance(data, str))
        return dataset

    def build_optimizer(self, trn, **kwargs):
        # noinspection PyCallByClass,PyTypeChecker
        return TransformerComponent.build_optimizer(self, trn, **kwargs)

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs):
        return MetricDict({
            'lemmas': CategoricalAccuracy(),
            'upos': CategoricalAccuracy(),
            'deps': AttachmentScore(),
            'feats': CategoricalAccuracy(),
        })

    def evaluate_dataloader(self,
                            data: DataLoader,
                            criterion: Callable,
                            metric: MetricDict = None,
                            output=False,
                            logger=None,
                            ratio_width=None,
                            **kwargs):

        metric.reset()
        self.model.eval()
        timer = CountdownTimer(len(data))
        total_loss = 0
        for idx, batch in enumerate(data):
            out, mask = self.feed_batch(batch)
            loss = out['loss']
            total_loss += loss.item()
            self.decode_output(out, mask, batch)
            self.update_metrics(metric, batch, out, mask)
            report = f'loss: {total_loss / (idx + 1):.4f} {metric.cstr()}'
            timer.log(report, logger=logger, ratio_percentage=False, ratio_width=ratio_width)
            del loss
            del out
            del mask
        return total_loss / len(data), metric

    # noinspection PyMethodOverriding
    def build_model(self,
                    transformer: ContextualWordEmbedding,
                    n_mlp_arc,
                    n_mlp_rel,
                    mlp_dropout,
                    mix_embedding,
                    layer_dropout,
                    training=True,
                    **kwargs) -> torch.nn.Module:
        assert bool(transformer.scalar_mix) == bool(mix_embedding), 'transformer.scalar_mix has to be 1 ' \
                                                                    'when mix_embedding is non-zero.'
        # noinspection PyTypeChecker
        return UniversalDependenciesModel(transformer.module(training=training),
                                          n_mlp_arc,
                                          n_mlp_rel,
                                          mlp_dropout,
                                          len(self.vocabs.rel),
                                          len(self.vocabs.lemma),
                                          len(self.vocabs.pos),
                                          len(self.vocabs.feat),
                                          mix_embedding,
                                          layer_dropout)

    def predict(self, data: Union[List[str], List[List[str]]], batch_size: int = None, **kwargs):
        if not data:
            return []
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        samples = self.build_samples(data)
        if not batch_size:
            batch_size = self.config.batch_size
        dataloader = self.build_dataloader(samples,
                                           device=self.devices[0], shuffle=False,
                                           **merge_dict(self.config,
                                                        batch_size=batch_size,
                                                        overwrite=True,
                                                        **kwargs))
        order = []
        outputs = []
        for batch in dataloader:
            out, mask = self.feed_batch(batch)
            self.decode_output(out, mask, batch)
            outputs.extend(self.prediction_to_human(out, batch))
            order.extend(batch[IDX])
        outputs = reorder(outputs, order)
        if flat:
            return outputs[0]
        return outputs

    def build_samples(self, data: List[List[str]]):
        return [{'FORM': x} for x in data]

    def fit(self,
            trn_data,
            dev_data,
            save_dir,
            transformer: ContextualWordEmbedding,
            sampler_builder=None,
            mix_embedding: int = 13,
            layer_dropout: int = 0.1,
            n_mlp_arc=768,
            n_mlp_rel=256,
            mlp_dropout=.33,
            lr=1e-3,
            transformer_lr=2.5e-5,
            patience=0.1,
            batch_size=32,
            epochs=30,
            gradient_accumulation=1,
            adam_epsilon=1e-8,
            weight_decay=0,
            warmup_steps=0.1,
            grad_norm=1.0,
            tree=False,
            proj=False,
            punct=False,
            logger=None,
            verbose=True,
            devices: Union[float, int, List[int]] = None, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

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
            loss, dev_metric = self.evaluate_dataloader(dev, criterion, metric, logger=logger, ratio_width=ratio_width)
            timer.update()
            report = f"{timer.elapsed_human} / {timer.total_time_human} ETA: {timer.eta_human}"
            if dev_metric > best_metric:
                best_epoch, best_metric = epoch, deepcopy(dev_metric)
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
        logger.info(f"Max score of dev is {best_metric.cstr()} at epoch {best_epoch}")
        logger.info(f"Average time of each epoch is {timer.elapsed_average_human}")
        logger.info(f"{timer.elapsed_human} elapsed")

    # noinspection PyMethodOverriding
    def fit_dataloader(self,
                       trn: DataLoader,
                       criterion,
                       optimizer,
                       metric: MetricDict,
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
            loss = out['loss']
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if eval_trn:
                self.decode_output(out, mask, batch)
                self.update_metrics(metric, batch, out, mask)
            if history.step(gradient_accumulation):
                self._step(optimizer, scheduler, grad_norm)
                report = f'loss: {total_loss / (idx + 1):.4f} {metric.cstr()}' if eval_trn \
                    else f'loss: {total_loss / (idx + 1):.4f}'
                timer.log(report, logger=logger, ratio_percentage=False, ratio_width=ratio_width)
            del loss
            del out
            del mask

    def decode_output(self, outputs, mask, batch):
        arc_scores, rel_scores = outputs['class_probabilities']['deps']['s_arc'], \
                                 outputs['class_probabilities']['deps']['s_rel']
        arc_preds, rel_preds = BiaffineDependencyParser.decode(self, arc_scores, rel_scores, mask, batch)
        outputs['arc_preds'], outputs['rel_preds'] = arc_preds, rel_preds
        return outputs

    def update_metrics(self, metrics, batch, outputs, mask):
        arc_preds, rel_preds, puncts = outputs['arc_preds'], outputs['rel_preds'], batch.get('punct_mask', None)
        BiaffineDependencyParser.update_metric(self, arc_preds, rel_preds, batch['arc'], batch['rel_id'], mask, puncts,
                                               metrics['deps'], batch)
        for task, key in zip(['lemmas', 'upos', 'feats'], ['lemma_id', 'pos_id', 'feat_id']):
            metric: Metric = metrics[task]
            pred = outputs['class_probabilities'][task]
            gold = batch[key]
            metric(pred.detach(), gold, mask=mask)
        return metrics

    def feed_batch(self, batch: dict):
        mask = self.compute_mask(batch)
        output_dict = self.model(batch, mask)
        if self.model.training:
            mask = mask.clone()
        mask[:, 0] = 0
        return output_dict, mask

    def compute_mask(self, batch):
        lens = batch['token_length']
        mask = lengths_to_mask(lens)
        return mask

    def _step(self, optimizer, scheduler, grad_norm):
        clip_grad_norm(self.model, grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    def input_is_flat(self, data):
        # noinspection PyCallByClass,PyTypeChecker
        return BiaffineDependencyParser.input_is_flat(self, data, False)

    def prediction_to_human(self, outputs: dict, batch):
        arcs, rels = outputs['arc_preds'], outputs['rel_preds']
        upos = outputs['class_probabilities']['upos'][:, 1:, :].argmax(-1).tolist()
        feats = outputs['class_probabilities']['feats'][:, 1:, :].argmax(-1).tolist()
        lemmas = outputs['class_probabilities']['lemmas'][:, 1:, :].argmax(-1).tolist()
        lem_vocab = self.vocabs['lemma'].idx_to_token
        pos_vocab = self.vocabs['pos'].idx_to_token
        feat_vocab = self.vocabs['feat'].idx_to_token
        # noinspection PyCallByClass,PyTypeChecker
        for tree, form, lemma, pos, feat in zip(BiaffineDependencyParser.prediction_to_head_rel(
                self, arcs, rels, batch), batch['token'], lemmas, upos, feats):
            form = form[1:]
            assert len(form) == len(tree)
            lemma = [apply_lemma_rule(t, lem_vocab[r]) for t, r in zip(form, lemma)]
            pos = [pos_vocab[x] for x in pos]
            feat = [feat_vocab[x] for x in feat]
            yield CoNLLSentence(
                [CoNLLUWord(id=i + 1, form=fo, lemma=l, upos=p, feats=fe, head=a, deprel=r) for
                 i, (fo, (a, r), l, p, fe) in enumerate(zip(form, tree, lemma, pos, feat))])

    def __call__(self, data, batch_size=None, **kwargs) -> Union[CoNLLSentence, List[CoNLLSentence]]:
        return super().__call__(data, batch_size, **kwargs)
