# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-08 20:51
import os
from collections import Counter
from typing import Union, Any, List

from hanlp.layers.transformers.pt_imports import AutoTokenizer, PreTrainedTokenizer, AutoModel_
import torch
from alnlp.modules.util import lengths_to_mask
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from hanlp_common.constant import ROOT, UNK, IDX
from hanlp.common.dataset import PadSequenceDataLoader
from hanlp.common.structure import History
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import LowerCase, FieldLength, PunctuationMask
from hanlp.common.vocab import Vocab
from hanlp.components.parsers.alg import decode_dep
from hanlp.components.parsers.biaffine.biaffine_model import BiaffineDependencyModel
from hanlp_common.conll import CoNLLWord, CoNLLSentence
from hanlp.datasets.parsing.conll_dataset import CoNLLParsingDataset, append_bos
from hanlp.layers.embeddings.util import index_word2vec_with_vocab
from hanlp.layers.transformers.utils import build_optimizer_scheduler_with_transformer
from hanlp.metrics.parsing.attachmentscore import AttachmentScore
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.util import isdebugging, merge_locals_kwargs, merge_dict, reorder


class BiaffineDependencyParser(TorchComponent):
    def __init__(self) -> None:
        """Biaffine dependency parsing (:cite:`dozat:17a`).
        """
        super().__init__()
        self.model: BiaffineDependencyModel = None
        self.transformer_tokenizer: PreTrainedTokenizer = None

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
        pos_key = 'CPOS' if 'CPOS' in self.vocabs else 'UPOS'
        for idx, each in enumerate(data):
            sample = {IDX: idx}
            if use_pos:
                token, pos = zip(*each)
                sample.update({'FORM': list(token), pos_key: list(pos)})
            else:
                token = each
                sample.update({'FORM': list(token)})
            samples.append(sample)
        return samples

    def input_is_flat(self, data, use_pos=None):
        if use_pos is None:
            use_pos = 'CPOS' in self.vocabs
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

    @property
    def use_pos(self):
        return self.config.get('feat', None) == 'pos'

    def fit(self, trn_data, dev_data, save_dir,
            feat=None,
            n_embed=100,
            pretrained_embed=None,
            transformer=None,
            average_subwords=False,
            word_dropout=0.2,
            transformer_hidden_dropout=None,
            layer_dropout=0,
            scalar_mix: int = None,
            embed_dropout=.33,
            n_lstm_hidden=400,
            n_lstm_layers=3,
            hidden_dropout=.33,
            n_mlp_arc=500,
            n_mlp_rel=100,
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
            separate_optimizer=False,
            patience=100,
            lowercase=False,
            epochs=50000,
            tree=False,
            proj=False,
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
            secondary_encoder=None,
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
                                gradient_accumulation=gradient_accumulation)
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
        if self.config.transformer:
            transformer = model.encoder.transformer
            optimizer = Adam(set(model.parameters()) - set(transformer.parameters()),
                             config.lr,
                             (config.mu, config.nu),
                             config.epsilon)
            if self.config.transformer_lr:
                num_training_steps = len(trn) * epochs // gradient_accumulation
                if self.config.separate_optimizer:
                    transformer_optimizer, transformer_scheduler = \
                        build_optimizer_scheduler_with_transformer(transformer,
                                                                   transformer,
                                                                   config.transformer_lr,
                                                                   config.transformer_lr,
                                                                   num_training_steps,
                                                                   config.warmup_steps,
                                                                   config.weight_decay,
                                                                   adam_epsilon=1e-8)
                else:
                    optimizer, scheduler = build_optimizer_scheduler_with_transformer(model,
                                                                                      transformer,
                                                                                      config.lr,
                                                                                      config.transformer_lr,
                                                                                      num_training_steps,
                                                                                      config.warmup_steps,
                                                                                      config.weight_decay,
                                                                                      adam_epsilon=1e-8)
                    transformer_optimizer, transformer_scheduler = None, None
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
        return optimizer, scheduler, transformer_optimizer, transformer_scheduler

    def build_transformer_tokenizer(self):
        transformer = self.config.transformer
        if transformer:
            transformer_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(transformer, use_fast=True)
        else:
            transformer_tokenizer = None
        self.transformer_tokenizer = transformer_tokenizer
        return transformer_tokenizer

    # noinspection PyMethodOverriding
    def build_dataloader(self,
                         data,
                         shuffle,
                         device,
                         training=False,
                         logger=None,
                         gradient_accumulation=1,
                         sampler_builder=None,
                         batch_size=None,
                         **kwargs) -> DataLoader:
        dataset = self.build_dataset(data)
        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger, self.config.transformer)
        transformer_tokenizer = self.transformer_tokenizer
        if transformer_tokenizer:
            dataset.transform.append(self.build_tokenizer_transform())
        dataset.append_transform(FieldLength('token', 'sent_length'))
        if isinstance(data, str):
            dataset.purge_cache()
        if len(dataset) > 1000 and isinstance(data, str):
            timer = CountdownTimer(len(dataset))
            self.cache_dataset(dataset, timer, training, logger)
        if self.config.transformer:
            lens = [len(sample['input_ids']) for sample in dataset]
        else:
            lens = [sample['sent_length'] for sample in dataset]
        if sampler_builder:
            sampler = sampler_builder.build(lens, shuffle, gradient_accumulation)
        else:
            sampler = None
        loader = PadSequenceDataLoader(dataset=dataset,
                                       batch_sampler=sampler,
                                       batch_size=batch_size,
                                       num_workers=0 if isdebugging() else 2,
                                       pad=self.get_pad_dict(),
                                       device=device,
                                       vocabs=self.vocabs)
        return loader

    def cache_dataset(self, dataset, timer, training=False, logger=None):
        for each in dataset:
            timer.log('Preprocessing and caching samples [blink][yellow]...[/yellow][/blink]')

    def get_pad_dict(self):
        return {'arc': 0}

    def build_dataset(self, data, bos_transform=None):
        if not bos_transform:
            bos_transform = append_bos
        transform = [bos_transform]
        if self.config.get('transform', None):
            transform.append(self.config.transform)
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

    def build_vocabs(self, dataset, logger=None, transformer=None):
        rel_vocab = self.vocabs.get('rel', None)
        if rel_vocab is None:
            rel_vocab = Vocab(unk_token=None, pad_token=self.config.get('pad_rel', None))
            self.vocabs.put(rel=rel_vocab)
        if self.config.get('feat', None) == 'pos' or self.config.get('use_pos', False):
            self.vocabs['pos'] = Vocab(unk_token=None, pad_token=None)

        timer = CountdownTimer(len(dataset))
        if transformer:
            token_vocab = None
        else:
            token_vocab = Vocab()
            self.vocabs.token = token_vocab
            unk = self.config.get('unk', None)
            if unk is not None:
                token_vocab.unk_token = unk
        if token_vocab and self.config.get('min_freq', None):
            counter = Counter()
            for sample in dataset:
                for form in sample['token']:
                    counter[form] += 1
            reserved_token = [token_vocab.pad_token, token_vocab.unk_token]
            if ROOT in token_vocab:
                reserved_token.append(ROOT)
            freq_words = reserved_token + [token for token, freq in counter.items() if
                                           freq >= self.config.min_freq]
            token_vocab.token_to_idx.clear()
            for word in freq_words:
                token_vocab(word)
        else:
            for i, sample in enumerate(dataset):
                timer.log('vocab building [blink][yellow]...[/yellow][/blink]', ratio_percentage=True)
        rel_vocab.set_unk_as_safe_unk()  # Some relation in dev set is OOV
        self.vocabs.lock()
        self.vocabs.summary(logger=logger)
        if token_vocab:
            self.config.n_words = len(self.vocabs['token'])
        if 'pos' in self.vocabs:
            self.config.n_feats = len(self.vocabs['pos'])
            self.vocabs['pos'].set_unk_as_safe_unk()
        self.config.n_rels = len(self.vocabs['rel'])
        if token_vocab:
            self.config.pad_index = self.vocabs['token'].pad_idx
            self.config.unk_index = self.vocabs['token'].unk_idx

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        pretrained_embed, transformer = self.build_embeddings(training=training)
        if pretrained_embed is not None:
            self.config.n_embed = pretrained_embed.size(-1)
        model = self.create_model(pretrained_embed, transformer)
        return model

    def create_model(self, pretrained_embed, transformer):
        return BiaffineDependencyModel(self.config,
                                       pretrained_embed,
                                       transformer,
                                       self.transformer_tokenizer)

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
                       **kwargs):
        self.model.train()

        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation))
        metric = self.build_metric(training=True)
        total_loss = 0
        for idx, batch in enumerate(trn):
            arc_scores, rel_scores, mask, puncts = self.feed_batch(batch)
            arcs, rels = batch['arc'], batch['rel_id']
            loss = self.compute_loss(arc_scores, rel_scores, arcs, rels, mask, criterion, batch)
            if gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask, batch)
            self.update_metric(arc_preds, rel_preds, arcs, rels, mask, puncts, metric, batch)
            if history.step(gradient_accumulation):
                self._step(optimizer, scheduler, transformer_optimizer, transformer_scheduler)
                report = self._report(total_loss / (timer.current + 1), metric)
                timer.log(report, ratio_percentage=False, logger=logger)
            del loss

    def _step(self, optimizer, scheduler, transformer_optimizer, transformer_scheduler):
        if self.config.get('grad_norm', None):
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.config.grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if self.config.transformer and self.config.transformer_lr and transformer_optimizer:
            transformer_optimizer.step()
            transformer_optimizer.zero_grad()
            transformer_scheduler.step()

    def feed_batch(self, batch):
        words, feats, lens, puncts = batch.get('token_id', None), batch.get('pos_id', None), batch['sent_length'], \
                                     batch.get('punct_mask', None)
        mask = lengths_to_mask(lens)
        arc_scores, rel_scores = self.model(words=words, feats=feats, mask=mask, batch=batch, **batch)
        # ignore the first token of each sentence
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        if self.model.training:
            mask = mask.clone()
        mask[:, 0] = 0
        return arc_scores, rel_scores, mask, puncts

    def _report(self, loss, metric: AttachmentScore):
        return f'loss: {loss:.4f} {metric}'

    def compute_loss(self, arc_scores, rel_scores, arcs, rels, mask, criterion, batch=None):
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        arc_loss = criterion(arc_scores, arcs)
        rel_loss = criterion(rel_scores, rels)
        loss = arc_loss + rel_loss

        return loss

    # noinspection PyUnboundLocalVariable
    @torch.no_grad()
    def evaluate_dataloader(self, loader: PadSequenceDataLoader, criterion, logger=None, filename=None, output=False,
                            ratio_width=None,
                            metric=None,
                            **kwargs):
        self.model.eval()

        loss = 0
        if not metric:
            metric = self.build_metric()
        if output:
            fp = open(output, 'w')
            predictions, build_data, data, order = self.before_outputs(None)

        timer = CountdownTimer(len(loader))
        use_pos = self.use_pos
        for batch in loader:
            arc_scores, rel_scores, mask, puncts = self.feed_batch(batch)
            if output:
                self.collect_outputs(arc_scores, rel_scores, mask, batch, predictions, order, data, use_pos,
                                     build_data)
            arcs, rels = batch['arc'], batch['rel_id']
            loss += self.compute_loss(arc_scores, rel_scores, arcs, rels, mask, criterion, batch).item()
            arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask, batch)
            self.update_metric(arc_preds, rel_preds, arcs, rels, mask, puncts, metric, batch)
            report = self._report(loss / (timer.current + 1), metric)
            if filename:
                report = f'{os.path.basename(filename)} ' + report
            timer.log(report, ratio_percentage=False, logger=logger, ratio_width=ratio_width)
        loss /= len(loader)
        if output:
            outputs = self.post_outputs(predictions, data, order, use_pos, build_data)
            for each in outputs:
                fp.write(f'{each}\n\n')
            fp.close()
            logger.info(f'Predictions saved in [underline][yellow]{output}[/yellow][/underline]')

        return loss, metric

    def update_metric(self, arc_preds, rel_preds, arcs, rels, mask, puncts, metric, batch=None):
        # ignore all punctuation if not specified
        if not self.config.punct:
            mask &= puncts
        metric(arc_preds, rel_preds, arcs, rels, mask)

    def decode(self, arc_scores, rel_scores, mask, batch=None):
        tree, proj = self.config.tree, self.config.get('proj', False)
        if tree:
            arc_preds = decode_dep(arc_scores, mask, tree, proj)
        else:
            arc_preds = arc_scores.argmax(-1)
        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    def build_criterion(self, **kwargs):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def build_metric(self, **kwargs):
        return AttachmentScore()

    def on_config_ready(self, **kwargs):
        self.build_transformer_tokenizer()  # We have to build tokenizer before building the dataloader and model
        self.config.patience = min(self.config.patience, self.config.epochs)

    def prediction_to_head_rel(self, arcs: torch.LongTensor, rels: torch.LongTensor, batch: dict):
        arcs = arcs[:, 1:]  # Skip the ROOT
        rels = rels[:, 1:]
        arcs = arcs.tolist()
        rels = rels.tolist()
        vocab = self.vocabs['rel'].idx_to_token
        for arcs_per_sent, rels_per_sent, tokens in zip(arcs, rels, batch['token']):
            tokens = tokens[1:]
            sent_len = len(tokens)
            result = list(zip(arcs_per_sent[:sent_len], [vocab[r] for r in rels_per_sent[:sent_len]]))
            yield result
