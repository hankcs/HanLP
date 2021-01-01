# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-15 20:55
import logging
from typing import Union, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from hanlp.common.dataset import PadSequenceDataLoader, SamplerBuilder, TransformableDataset
from hanlp.common.structure import History
from hanlp.common.transform import FieldLength, TransformList
from hanlp.common.vocab import Vocab
from hanlp.components.classifiers.transformer_classifier import TransformerComponent
from hanlp.components.taggers.tagger import Tagger
from hanlp.datasets.ner.tsv import TSVTaggingDataset
from hanlp.layers.crf.crf import CRF
from hanlp.layers.transformers.encoder import TransformerEncoder
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp.utils.time_util import CountdownTimer
from hanlp.utils.torch_util import clip_grad_norm
from hanlp_common.util import merge_locals_kwargs
from alnlp.modules.util import lengths_to_mask


# noinspection PyAbstractClass
class TransformerTaggingModel(nn.Module):
    def __init__(self,
                 encoder: TransformerEncoder,
                 num_labels,
                 crf=False,
                 secondary_encoder=None) -> None:
        """
        A shallow tagging model use transformer as decoder.
        Args:
            encoder: A pretrained transformer.
            num_labels: Size of tagset.
            crf: True to enable CRF.
            crf_constraints: The allowed transitions (from_label_id, to_label_id).
        """
        super().__init__()
        self.encoder = encoder
        self.secondary_encoder = secondary_encoder
        # noinspection PyUnresolvedReferences
        self.classifier = nn.Linear(encoder.transformer.config.hidden_size, num_labels)
        self.crf = CRF(num_labels) if crf else None

    def forward(self, lens: torch.LongTensor, input_ids, token_span, token_type_ids=None):
        mask = lengths_to_mask(lens)
        x = self.encoder(input_ids, token_span=token_span, token_type_ids=token_type_ids)
        if self.secondary_encoder:
            x = self.secondary_encoder(x, mask=mask)
        x = self.classifier(x)
        return x, mask


class TransformerTagger(TransformerComponent, Tagger):

    def __init__(self, **kwargs) -> None:
        """A simple tagger using a linear layer with an optional CRF (:cite:`lafferty2001conditional`) layer for
        any tagging tasks including PoS tagging and many others.

        Args:
            **kwargs: Not used.
        """
        super().__init__(**kwargs)
        self._tokenizer_transform = None
        self.model: TransformerTaggingModel = None

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
                       transformer_grad_norm=None,
                       teacher: Tagger = None,
                       kd_criterion=None,
                       temperature_scheduler=None,
                       ratio_width=None,
                       **kwargs):
        optimizer, scheduler = optimizer
        if teacher:
            scheduler, lambda_scheduler = scheduler
        else:
            lambda_scheduler = None
        self.model.train()
        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation=gradient_accumulation))
        total_loss = 0
        for idx, batch in enumerate(trn):
            out, mask = self.feed_batch(batch)
            y = batch['tag_id']
            loss = self.compute_loss(criterion, out, y, mask)
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            if teacher:
                with torch.no_grad():
                    out_T, _ = teacher.feed_batch(batch)
                # noinspection PyNoneFunctionAssignment
                kd_loss = self.compute_distill_loss(kd_criterion, out, out_T, mask, temperature_scheduler)
                _lambda = float(lambda_scheduler)
                loss = _lambda * loss + (1 - _lambda) * kd_loss
            loss.backward()
            total_loss += loss.item()
            prediction = self.decode_output(out, mask, batch)
            self.update_metrics(metric, out, y, mask, batch, prediction)
            if history.step(gradient_accumulation):
                self._step(optimizer, scheduler, grad_norm, transformer_grad_norm, lambda_scheduler)
                report = f'loss: {total_loss / (idx + 1):.4f} {metric}'
                timer.log(report, logger=logger, ratio_percentage=False, ratio_width=ratio_width)
            del loss
            del out
            del mask

    def _step(self, optimizer, scheduler, grad_norm, transformer_grad_norm, lambda_scheduler):
        clip_grad_norm(self.model, grad_norm, self.model.encoder.transformer, transformer_grad_norm)
        optimizer.step()
        scheduler.step()
        if lambda_scheduler:
            lambda_scheduler.step()
        optimizer.zero_grad()

    def compute_distill_loss(self, kd_criterion, out_S, out_T, mask, temperature_scheduler):
        logits_S = out_S[mask]
        logits_T = out_T[mask]
        temperature = temperature_scheduler(logits_S, logits_T)
        return kd_criterion(logits_S, logits_T, temperature)

    def build_model(self, **kwargs) -> torch.nn.Module:
        model = TransformerTaggingModel(self.build_transformer(),
                                        len(self.vocabs.tag),
                                        self.config.crf,
                                        self.config.get('secondary_encoder', None),
                                        )
        return model

    # noinspection PyMethodOverriding
    def build_dataloader(self, data, batch_size, shuffle, device, logger: logging.Logger = None,
                         sampler_builder: SamplerBuilder = None, gradient_accumulation=1, **kwargs) -> DataLoader:
        if isinstance(data, TransformableDataset):
            dataset = data
        else:
            args = dict((k, self.config.get(k, None)) for k in
                        ['delimiter', 'max_seq_len', 'sent_delimiter', 'char_level', 'hard_constraint'])
            dataset = self.build_dataset(data, **args)
        if self.config.token_key is None:
            self.config.token_key = next(iter(dataset[0]))
            logger.info(
                f'Guess [bold][blue]token_key={self.config.token_key}[/blue][/bold] according to the '
                f'training dataset: [blue]{dataset}[/blue]')
        dataset.append_transform(self.tokenizer_transform)
        dataset.append_transform(self.last_transform())
        if not isinstance(data, list):
            dataset.purge_cache()
        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger)
        if sampler_builder is not None:
            sampler = sampler_builder.build([len(x[f'{self.config.token_key}_input_ids']) for x in dataset], shuffle,
                                            gradient_accumulation=gradient_accumulation if shuffle else 1)
        else:
            sampler = None
        return PadSequenceDataLoader(dataset, batch_size, shuffle, device=device, batch_sampler=sampler)

    def build_dataset(self, data, transform=None, **kwargs):
        return TSVTaggingDataset(data, transform=transform, **kwargs)

    def last_transform(self):
        return TransformList(self.vocabs, FieldLength(self.config.token_key))

    @property
    def tokenizer_transform(self) -> TransformerSequenceTokenizer:
        if not self._tokenizer_transform:
            self._tokenizer_transform = TransformerSequenceTokenizer(self.transformer_tokenizer,
                                                                     self.config.token_key,
                                                                     ret_token_span=True)
        return self._tokenizer_transform

    def build_vocabs(self, trn, logger, **kwargs):
        self.vocabs.tag = Vocab(pad_token=None, unk_token=None)
        timer = CountdownTimer(len(trn))
        max_seq_len = 0
        token_key = self.config.token_key
        for each in trn:
            max_seq_len = max(max_seq_len, len(each[token_key]))
            timer.log(f'Building vocab [blink][yellow]...[/yellow][/blink] (longest sequence: {max_seq_len})')
        self.vocabs.tag.set_unk_as_safe_unk()
        self.vocabs.lock()
        self.vocabs.summary(logger)

    # noinspection PyMethodOverriding
    def fit(self,
            trn_data,
            dev_data,
            save_dir,
            transformer,
            average_subwords=False,
            word_dropout: float = 0.2,
            hidden_dropout=None,
            layer_dropout=0,
            scalar_mix=None,
            mix_embedding: int = 0,
            grad_norm=5.0,
            transformer_grad_norm=None,
            lr=5e-5,
            transformer_lr=None,
            transformer_layers=None,
            gradient_accumulation=1,
            adam_epsilon=1e-6,
            weight_decay=0,
            warmup_steps=0.1,
            secondary_encoder=None,
            crf=False,
            reduction='sum',
            batch_size=32,
            sampler_builder: SamplerBuilder = None,
            epochs=3,
            patience=5,
            token_key=None,
            max_seq_len=None, sent_delimiter=None, char_level=False, hard_constraint=False,
            transform=None,
            logger=None,
            devices: Union[float, int, List[int]] = None,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def feed_batch(self, batch: dict):
        features = [batch[k] for k in self.tokenizer_transform.output_key]
        if len(features) == 2:
            input_ids, token_span = features
        else:
            input_ids, token_span = features[0], None
        lens = batch[f'{self.config.token_key}_length']
        x, mask = self.model(lens, input_ids, token_span, batch.get(f'{self.config.token_key}_token_type_ids'))
        return x, mask

    # noinspection PyMethodOverriding
    def distill(self,
                teacher: str,
                trn_data,
                dev_data,
                save_dir,
                transformer: str,
                batch_size=None,
                temperature_scheduler='flsw',
                epochs=None,
                devices=None,
                logger=None,
                seed=None,
                **kwargs):
        return super().distill(**merge_locals_kwargs(locals(), kwargs))
