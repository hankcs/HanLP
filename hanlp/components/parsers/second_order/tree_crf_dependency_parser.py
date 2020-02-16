# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-08 20:51
import functools
import os
from typing import Union, Any, List

import torch
from alnlp.modules.util import lengths_to_mask
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from hanlp_common.constant import UNK, IDX
from hanlp.common.dataset import PadSequenceDataLoader
from hanlp.common.structure import History
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import LowerCase, FieldLength, PunctuationMask, TransformList
from hanlp.common.vocab import Vocab, VocabCounter
from hanlp_common.conll import CoNLLWord, CoNLLSentence
from hanlp.components.parsers.constituency.treecrf import CRF2oDependency
from hanlp.components.parsers.second_order.model import DependencyModel
from hanlp.components.parsers.second_order.treecrf_decoder import TreeCRFDecoder
from hanlp.datasets.parsing.conll_dataset import CoNLLParsingDataset, append_bos, get_sibs
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding, ContextualWordEmbeddingModule
from hanlp.layers.embeddings.embedding import Embedding, EmbeddingList, ConcatModuleList
from hanlp.layers.embeddings.util import index_word2vec_with_vocab
from hanlp.layers.transformers.pt_imports import AutoModel_
from hanlp.layers.transformers.utils import build_optimizer_scheduler_with_transformer
from hanlp.metrics.parsing.attachmentscore import AttachmentScore
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.util import merge_locals_kwargs, merge_dict, reorder


class TreeConditionalRandomFieldDependencyParser(TorchComponent):
    def __init__(self) -> None:
        super().__init__()
        self.model: DependencyModel = self.model
        self._transformer_transform = None

    def predict(self, data: Any, batch_size=None, batch_max_tokens=None, output_format='conllx', **kwargs):
        if not data:
            return []
        use_pos = self.use_pos
        flat = self.input_is_flat(data, use_pos)
        if flat:
            data = [data]
        samples = self.build_samples(data, use_pos)
        if not batch_max_tokens:
            batch_max_tokens = self.config.batch_max_tokens
        if not batch_size:
            batch_size = self.config.batch_size
        dataloader = self.build_dataloader(samples,
                                           device=self.devices[0], shuffle=False,
                                           **merge_dict(self.config,
                                                        batch_size=batch_size,
                                                        batch_max_tokens=batch_max_tokens,
                                                        overwrite=True,
                                                        **kwargs))
        predictions, build_data, data, order = self.before_outputs(data)
        for batch in dataloader:
            arc_scores, rel_scores, mask, puncts = self.feed_batch(batch)
            self.collect_outputs(arc_scores, rel_scores, mask, batch, predictions, order, data, use_pos,
                                 build_data)
        outputs = self.post_outputs(predictions, data, order, use_pos, build_data)
        if flat:
            return outputs[0]
        return outputs

    def build_samples(self, data, use_pos=None):
        samples = []
        for idx, each in enumerate(data):
            sample = {IDX: idx}
            if use_pos:
                token, pos = zip(*each)
                sample.update({'FORM': list(token), 'CPOS': list(pos)})
            else:
                token = each
                sample.update({'FORM': list(token)})
            samples.append(sample)
        return samples

    def input_is_flat(self, data, use_pos=None):
        if use_pos:
            flat = isinstance(data[0], (list, tuple)) and isinstance(data[0][0], str)
        else:
            flat = isinstance(data[0], str)
        return flat

    def before_outputs(self, data):
        predictions, order = [], []
        build_data = data is None
        if build_data:
            data = []
        return predictions, build_data, data, order

    def post_outputs(self, predictions, data, order, use_pos, build_data):
        predictions = reorder(predictions, order)
        if build_data:
            data = reorder(data, order)
        outputs = []
        self.predictions_to_human(predictions, outputs, data, use_pos)
        return outputs

    def predictions_to_human(self, predictions, outputs, data, use_pos):
        for d, (arcs, rels) in zip(data, predictions):
            sent = CoNLLSentence()
            for idx, (cell, a, r) in enumerate(zip(d, arcs, rels)):
                if use_pos:
                    token, pos = cell
                else:
                    token, pos = cell, None
                sent.append(CoNLLWord(idx + 1, token, cpos=pos, head=a, deprel=self.vocabs['rel'][r]))
            outputs.append(sent)

    def collect_outputs(self, arc_scores, rel_scores, mask, batch, predictions, order, data, use_pos,
                        build_data):
        lens = [len(token) - 1 for token in batch['token']]
        arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask, batch)
        self.collect_outputs_extend(predictions, arc_preds, rel_preds, lens, mask)
        order.extend(batch[IDX])
        if build_data:
            if use_pos:
                data.extend(zip(batch['FORM'], batch['CPOS']))
            else:
                data.extend(batch['FORM'])

    def collect_outputs_extend(self, predictions: list, arc_preds, rel_preds, lens, mask):
        predictions.extend(zip([seq.tolist() for seq in arc_preds[mask].split(lens)],
                               [seq.tolist() for seq in rel_preds[mask].split(lens)]))

    def fit(self,
            trn_data,
            dev_data,
            save_dir,
            embed,
            n_mlp_arc=500,
            n_mlp_rel=100,
            n_mlp_sib=100,
            mlp_dropout=.33,
            lr=2e-3,
            transformer_lr=5e-5,
            mu=.9,
            nu=.9,
            epsilon=1e-12,
            grad_norm=5.0,
            decay=.75,
            decay_steps=5000,
            weight_decay=0,
            warmup_steps=0.1,
            separate_optimizer=True,
            patience=100,
            lowercase=False,
            epochs=50000,
            tree=False,
            proj=True,
            mbr=True,
            partial=False,
            punct=False,
            min_freq=2,
            logger=None,
            verbose=True,
            unk=UNK,
            max_sequence_length=512,
            batch_size=None,
            sampler_builder=None,
            gradient_accumulation=1,
            devices: Union[float, int, List[int]] = None,
            transform=None,
            eval_trn=False,
            bos='\0',
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def execute_training_loop(self, trn, dev, devices, epochs, logger, patience, save_dir, optimizer,
                              gradient_accumulation, **kwargs):
        optimizer, scheduler, transformer_optimizer, transformer_scheduler = optimizer
        criterion = self.build_criterion()
        best_e, best_metric = 0, self.build_metric()
        timer = CountdownTimer(epochs)
        history = History()
        ratio_width = len(f'{len(trn) // gradient_accumulation}/{len(trn) // gradient_accumulation}')
        for epoch in range(1, epochs + 1):
            # train one epoch and update the parameters
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, optimizer, scheduler, criterion, epoch, logger, history,
                                transformer_optimizer, transformer_scheduler,
                                gradient_accumulation=gradient_accumulation, eval_trn=self.config.eval_trn)
            loss, dev_metric = self.evaluate_dataloader(dev, criterion, ratio_width=ratio_width, logger=logger)
            timer.update()
            # logger.info(f"{'Dev' + ' ' * ratio_width} loss: {loss:.4f} {dev_metric}")
            # save the model if it is the best so far
            report = f"{timer.elapsed_human} / {timer.total_time_human} ETA: {timer.eta_human}"
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                self.save_weights(save_dir)
                report += ' ([red]saved[/red])'
            else:
                if patience != epochs:
                    report += f' ({epoch - best_e}/{patience})'
                else:
                    report += f' ({epoch - best_e})'
            logger.info(report)
            if patience is not None and epoch - best_e >= patience:
                logger.info(f'LAS has stopped improving for {patience} epochs, early stop.')
                break
        timer.stop()
        if not best_e:
            self.save_weights(save_dir)
        elif best_e != epoch:
            self.load_weights(save_dir)
        logger.info(f"Max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        logger.info(f"Average time of each epoch is {timer.elapsed_average_human}")
        logger.info(f"{timer.elapsed_human} elapsed")

    def build_optimizer(self, epochs, trn, gradient_accumulation, **kwargs):
        config = self.config
        model = self.model
        if isinstance(model, nn.DataParallel):
            model = model.module
        transformer = self._get_transformer_builder()
        if transformer and transformer.trainable:
            transformer = self._get_transformer()
            optimizer = Adam(set(model.parameters()) - set(transformer.parameters()),
                             config.lr,
                             (config.mu, config.nu),
                             config.epsilon)
            if self.config.transformer_lr:
                num_training_steps = len(trn) * epochs // gradient_accumulation
                if not self.config.separate_optimizer:
                    optimizer, scheduler = build_optimizer_scheduler_with_transformer(model,
                                                                                      transformer,
                                                                                      config.lr,
                                                                                      config.transformer_lr,
                                                                                      num_training_steps,
                                                                                      config.warmup_steps,
                                                                                      config.weight_decay,
                                                                                      config.epsilon)
                    transformer_optimizer, transformer_scheduler = None, None
                else:
                    transformer_optimizer, transformer_scheduler = \
                        build_optimizer_scheduler_with_transformer(transformer,
                                                                   transformer,
                                                                   config.lr,
                                                                   config.transformer_lr,
                                                                   num_training_steps,
                                                                   config.warmup_steps,
                                                                   config.weight_decay,
                                                                   config.epsilon)
            else:
                transformer.requires_grad_(False)
                transformer_optimizer, transformer_scheduler = None, None
        else:
            optimizer = Adam(model.parameters(),
                             config.lr,
                             (config.mu, config.nu),
                             config.epsilon)
            transformer_optimizer, transformer_scheduler = None, None
        if self.config.separate_optimizer:
            scheduler = ExponentialLR(optimizer, config.decay ** (1 / config.decay_steps))
        # noinspection PyUnboundLocalVariable
        optimizer = Adam(model.parameters(), **{'lr': 0.002, 'betas': (0.9, 0.9), 'eps': 1e-12})
        scheduler = ExponentialLR(optimizer, **{'gamma': 0.9999424652406974})
        return optimizer, scheduler, transformer_optimizer, transformer_scheduler

    # noinspection PyMethodOverriding
    def build_dataloader(self,
                         data,
                         shuffle,
                         device,
                         embed: Embedding,
                         training=False,
                         logger=None,
                         gradient_accumulation=1,
                         sampler_builder=None,
                         batch_size=None,
                         bos='\0',
                         **kwargs) -> DataLoader:
        first_transform = TransformList(functools.partial(append_bos, bos=bos))
        embed_transform = embed.transform(vocabs=self.vocabs)
        transformer_transform = self._get_transformer_transform_from_transforms(embed_transform)
        if embed_transform:
            if transformer_transform and isinstance(embed_transform, TransformList):
                embed_transform.remove(transformer_transform)

            first_transform.append(embed_transform)
        dataset = self.build_dataset(data, first_transform=first_transform)
        if self.config.get('transform', None):
            dataset.append_transform(self.config.transform)

        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger, self._transformer_trainable())
        if transformer_transform and isinstance(embed_transform, TransformList):
            embed_transform.append(transformer_transform)

        dataset.append_transform(FieldLength('token', 'sent_length'))
        if isinstance(data, str):
            dataset.purge_cache()
        if len(dataset) > 1000 and isinstance(data, str):
            timer = CountdownTimer(len(dataset))
            self.cache_dataset(dataset, timer, training, logger)
        if sampler_builder:
            lens = [sample['sent_length'] for sample in dataset]
            sampler = sampler_builder.build(lens, shuffle, gradient_accumulation)
        else:
            sampler = None
        loader = PadSequenceDataLoader(dataset=dataset,
                                       batch_sampler=sampler,
                                       batch_size=batch_size,
                                       pad=self.get_pad_dict(),
                                       device=device,
                                       vocabs=self.vocabs)
        return loader

    def cache_dataset(self, dataset, timer, training=False, logger=None):
        for each in dataset:
            timer.log('Preprocessing and caching samples [blink][yellow]...[/yellow][/blink]')

    def get_pad_dict(self):
        return {'arc': 0}

    def build_dataset(self, data, first_transform=None):
        if not first_transform:
            first_transform = append_bos
        transform = [first_transform, get_sibs]
        if self.config.get('lowercase', False):
            transform.append(LowerCase('token'))
        transform.append(self.vocabs)
        if not self.config.punct:
            transform.append(PunctuationMask('token', 'punct_mask'))
        return CoNLLParsingDataset(data, transform=transform)

    def build_tokenizer_transform(self):
        return TransformerSequenceTokenizer(self.transformer_tokenizer, 'token', '',
                                            ret_token_span=True, cls_is_bos=True,
                                            max_seq_length=self.config.get('max_sequence_length',
                                                                           512),
                                            truncate_long_sequences=False)

    def build_vocabs(self, dataset, logger=None, transformer=False):
        rel_vocab = self.vocabs.get('rel', None)
        if rel_vocab is None:
            rel_vocab = Vocab(unk_token=None, pad_token=self.config.get('pad_rel', None))
            self.vocabs.put(rel=rel_vocab)

        timer = CountdownTimer(len(dataset))
        if transformer:
            token_vocab = None
        else:
            self.vocabs.token = token_vocab = VocabCounter(unk_token=self.config.get('unk', UNK))
        for i, sample in enumerate(dataset):
            timer.log('Building vocab [blink][yellow]...[/yellow][/blink]', ratio_percentage=True)
        min_freq = self.config.get('min_freq', None)
        if min_freq:
            token_vocab.trim(min_freq)
        rel_vocab.set_unk_as_safe_unk()  # Some relation in dev set is OOV
        self.vocabs.lock()
        self.vocabs.summary(logger=logger)
        if token_vocab:
            self.config.n_words = len(self.vocabs['token'])
        self.config.n_rels = len(self.vocabs['rel'])
        if token_vocab:
            self.config.pad_index = self.vocabs['token'].pad_idx
            self.config.unk_index = self.vocabs['token'].unk_idx

    # noinspection PyMethodOverriding
    def build_model(self, embed: Embedding, encoder, n_mlp_arc, n_mlp_rel, mlp_dropout, n_mlp_sib, training=True,
                    **kwargs) -> torch.nn.Module:
        model = DependencyModel(
            embed=embed.module(vocabs=self.vocabs),
            encoder=encoder,
            decoder=TreeCRFDecoder(encoder.get_output_dim(), n_mlp_arc, n_mlp_sib, n_mlp_rel, mlp_dropout,
                                   len(self.vocabs['rel']))
        )
        return model

    def build_embeddings(self, training=True):
        pretrained_embed = None
        if self.config.get('pretrained_embed', None):
            pretrained_embed = index_word2vec_with_vocab(self.config.pretrained_embed, self.vocabs['token'],
                                                         init='zeros', normalize=True)
        transformer = self.config.transformer
        if transformer:
            transformer = AutoModel_.from_pretrained(transformer, training=training)
        return pretrained_embed, transformer

    # noinspection PyMethodOverriding
    def fit_dataloader(self,
                       trn,
                       optimizer,
                       scheduler,
                       criterion,
                       epoch,
                       logger,
                       history: History,
                       transformer_optimizer=None,
                       transformer_scheduler=None,
                       gradient_accumulation=1,
                       eval_trn=False,
                       **kwargs):
        self.model.train()

        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation))
        metric = self.build_metric(training=True)
        total_loss = 0
        for idx, batch in enumerate(trn):
            optimizer.zero_grad()
            (s_arc, s_sib, s_rel), mask, puncts = self.feed_batch(batch)
            arcs, sibs, rels = batch['arc'], batch['sib_id'], batch['rel_id']

            loss, s_arc = self.compute_loss(s_arc, s_sib, s_rel, arcs, sibs, rels, mask)
            if gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if eval_trn:
                arc_preds, rel_preds = self.decode(s_arc, s_sib, s_rel, mask)
                self.update_metric(arc_preds, rel_preds, arcs, rels, mask, puncts, metric)
            if history.step(gradient_accumulation):
                self._step(optimizer, scheduler, transformer_optimizer, transformer_scheduler)
                report = self._report(total_loss / (timer.current + 1), metric if eval_trn else None)
                lr = scheduler.get_last_lr()[0]
                report += f' lr: {lr:.4e}'
                timer.log(report, ratio_percentage=False, logger=logger)
            del loss

    def _step(self, optimizer, scheduler, transformer_optimizer, transformer_scheduler):
        if self.config.get('grad_norm', None):
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.config.grad_norm)
        optimizer.step()
        scheduler.step()
        if self._transformer_transform and self.config.transformer_lr and transformer_optimizer:
            transformer_optimizer.step()
            transformer_optimizer.zero_grad()
            transformer_scheduler.step()

    def feed_batch(self, batch):
        words, feats, lens, puncts = batch.get('token_id', None), batch.get('pos_id', None), batch['sent_length'], \
                                     batch.get('punct_mask', None)
        mask = lengths_to_mask(lens)
        logits = self.model(batch, mask)
        if self.model.training:
            mask = mask.clone()
        # ignore the first token of each sentence
        mask[:, 0] = 0
        return logits, mask, puncts

    def _report(self, loss, metric: AttachmentScore = None):
        return f'loss: {loss:.4f} {metric}' if metric else f'loss: {loss:.4f}'

    def compute_loss(self, s_arc, s_sib, s_rel, arcs, sibs, rels, mask):
        crf: CRF2oDependency = self.model.decoder.crf
        return crf.loss(s_arc, s_sib, s_rel, arcs, sibs, rels, mask, self.config.mbr, self.config.partial)

    # noinspection PyUnboundLocalVariable
    @torch.no_grad()
    def evaluate_dataloader(self, loader: PadSequenceDataLoader, criterion, logger=None, filename=None, output=False,
                            ratio_width=None,
                            metric=None,
                            **kwargs):
        self.model.eval()

        total_loss = 0
        if not metric:
            metric = self.build_metric()

        timer = CountdownTimer(len(loader))
        for batch in loader:
            (s_arc, s_sib, s_rel), mask, puncts = self.feed_batch(batch)
            arcs, sibs, rels = batch['arc'], batch['sib_id'], batch['rel_id']
            loss, s_arc = self.compute_loss(s_arc, s_sib, s_rel, arcs, sibs, rels, mask)
            total_loss += float(loss)
            arc_preds, rel_preds = self.decode(s_arc, s_sib, s_rel, mask)
            self.update_metric(arc_preds, rel_preds, arcs, rels, mask, puncts, metric)
            report = self._report(total_loss / (timer.current + 1), metric)
            if filename:
                report = f'{os.path.basename(filename)} ' + report
            timer.log(report, ratio_percentage=False, logger=logger, ratio_width=ratio_width)
        total_loss /= len(loader)

        return total_loss, metric

    def update_metric(self, arc_preds, rel_preds, arcs, rels, mask, puncts, metric):
        # ignore all punctuation if not specified
        if not self.config.punct:
            mask &= puncts
        metric(arc_preds, rel_preds, arcs, rels, mask)

    def decode(self, s_arc, s_sib, s_rel, mask):
        crf: CRF2oDependency = self.model.decoder.crf
        return crf.decode(s_arc, s_sib, s_rel, mask, self.config.tree and not self.model.training, self.config.mbr,
                          self.config.proj)

    def build_criterion(self, **kwargs):
        return None

    def build_metric(self, **kwargs):
        return AttachmentScore()

    def _get_transformer_transform_from_transforms(self, transform: Union[
        TransformList, TransformerSequenceTokenizer]) -> TransformerSequenceTokenizer:
        def _get():
            if isinstance(transform, TransformerSequenceTokenizer):
                # noinspection PyTypeChecker
                return transform
            elif isinstance(transform, TransformList):
                # noinspection PyTypeChecker,PyArgumentList
                for each in transform:
                    if isinstance(each, TransformerSequenceTokenizer):
                        return each

        if self._transformer_transform is None:
            self._transformer_transform = _get()
        return self._transformer_transform

    def _get_transformer(self):
        embed = self.model.embed
        if isinstance(embed, ContextualWordEmbeddingModule):
            return embed
        if isinstance(embed, ConcatModuleList):
            for each in embed:
                if isinstance(each, ContextualWordEmbeddingModule):
                    return each

    def _get_transformer_builder(self):
        embed: Embedding = self.config.embed
        if isinstance(embed, ContextualWordEmbedding):
            return embed
        if isinstance(embed, EmbeddingList):
            for each in embed.to_list():
                if isinstance(embed, ContextualWordEmbedding):
                    return each

    def _transformer_trainable(self):
        builder = self._get_transformer_builder()
        if not builder:
            return False
        return builder.trainable
