# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-08 16:31
import logging
from abc import ABC
from typing import Callable, Union
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from hanlp_common.constant import IDX
from hanlp.common.dataset import TableDataset, SortingSampler, PadSequenceDataLoader, TransformableDataset
from hanlp.common.torch_component import TorchComponent
from hanlp.common.vocab import Vocab
from hanlp.components.distillation.schedulers import LinearTeacherAnnealingScheduler
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.layers.transformers.encoder import TransformerEncoder
from hanlp.layers.transformers.pt_imports import PreTrainedModel, AutoTokenizer, BertTokenizer
from hanlp.layers.transformers.utils import transformer_sliding_window, build_optimizer_scheduler_with_transformer
from hanlp.metrics.accuracy import CategoricalAccuracy
from hanlp.transform.transformer_tokenizer import TransformerTextTokenizer
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.util import merge_locals_kwargs, merge_dict, isdebugging


class TransformerClassificationModel(nn.Module):

    def __init__(self,
                 transformer: PreTrainedModel,
                 num_labels: int,
                 max_seq_length=512) -> None:
        super().__init__()
        self.max_seq_length = max_seq_length
        self.transformer = transformer
        self.dropout = nn.Dropout(transformer.config.hidden_dropout_prob)
        self.classifier = nn.Linear(transformer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        seq_length = input_ids.size(-1)
        if seq_length > self.max_seq_length:
            sequence_output = transformer_sliding_window(self.transformer, input_ids,
                                                         max_pieces=self.max_seq_length, ret_cls='max')
        else:
            sequence_output = self.transformer(input_ids, attention_mask, token_type_ids)[0][:, 0, :]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class TransformerComponent(TorchComponent, ABC):
    def __init__(self, **kwargs) -> None:
        """ The base class for transorfmer based components. If offers methods to build transformer tokenizers
        , optimizers and models.

        Args:
            **kwargs: Passed to config.
        """
        super().__init__(**kwargs)
        self.transformer_tokenizer = None

    def build_optimizer(self,
                        trn,
                        epochs,
                        lr,
                        adam_epsilon,
                        weight_decay,
                        warmup_steps,
                        transformer_lr=None,
                        teacher=None,
                        **kwargs):
        num_training_steps = len(trn) * epochs // self.config.get('gradient_accumulation', 1)
        if transformer_lr is None:
            transformer_lr = lr
        transformer = self.model.encoder.transformer
        optimizer, scheduler = build_optimizer_scheduler_with_transformer(self.model, transformer,
                                                                          lr, transformer_lr,
                                                                          num_training_steps, warmup_steps,
                                                                          weight_decay, adam_epsilon)
        if teacher:
            lambda_scheduler = LinearTeacherAnnealingScheduler(num_training_steps)
            scheduler = (scheduler, lambda_scheduler)
        return optimizer, scheduler

    def fit(self, trn_data, dev_data, save_dir,
            transformer=None,
            lr=5e-5,
            transformer_lr=None,
            adam_epsilon=1e-8,
            weight_decay=0,
            warmup_steps=0.1,
            batch_size=32,
            gradient_accumulation=1,
            grad_norm=5.0,
            transformer_grad_norm=None,
            average_subwords=False,
            scalar_mix: Union[ScalarMixWithDropoutBuilder, int] = None,
            word_dropout=None,
            hidden_dropout=None,
            max_sequence_length=None,
            ret_raw_hidden_states=False,
            batch_max_tokens=None,
            epochs=3,
            logger=None,
            devices: Union[float, int, List[int]] = None,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def on_config_ready(self, **kwargs):
        super().on_config_ready(**kwargs)
        if 'albert_chinese' in self.config.transformer:
            self.transformer_tokenizer = BertTokenizer.from_pretrained(self.config.transformer, use_fast=True)
        else:
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(self.config.transformer, use_fast=True)

    def build_transformer(self, training=True):
        transformer = TransformerEncoder(self.config.transformer, self.transformer_tokenizer,
                                         self.config.average_subwords,
                                         self.config.scalar_mix, self.config.word_dropout,
                                         self.config.max_sequence_length, self.config.ret_raw_hidden_states,
                                         training=training)
        transformer_layers = self.config.get('transformer_layers', None)
        if transformer_layers:
            transformer.transformer.encoder.layer = transformer.transformer.encoder.layer[:-transformer_layers]
        return transformer


class TransformerClassifier(TransformerComponent):

    def __init__(self, **kwargs) -> None:
        """A classifier using transformer as encoder.

        Args:
            **kwargs: Passed to config.
        """
        super().__init__(**kwargs)
        self.model: TransformerClassificationModel = None

    def build_criterion(self, **kwargs):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def build_metric(self, **kwargs):
        return CategoricalAccuracy()

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, **kwargs):
        best_epoch, best_metric = 0, -1
        timer = CountdownTimer(epochs)
        ratio_width = len(f'{len(trn)}/{len(trn)}')
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger)
            if dev:
                self.evaluate_dataloader(dev, criterion, metric, logger, ratio_width=ratio_width)
            report = f'{timer.elapsed_human}/{timer.total_time_human}'
            dev_score = metric.get_metric()
            if dev_score > best_metric:
                self.save_weights(save_dir)
                best_metric = dev_score
                report += ' [red]saved[/red]'
            timer.log(report, ratio_percentage=False, newline=True, ratio=False)

    @property
    def label_vocab(self):
        return self.vocabs[self.config.label_key]

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        self.model.train()
        timer = CountdownTimer(len(trn))
        optimizer, scheduler = optimizer
        total_loss = 0
        metric.reset()
        for batch in trn:
            optimizer.zero_grad()
            logits = self.feed_batch(batch)
            target = batch['label_id']
            loss = self.compute_loss(criterion, logits, target, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            self.update_metric(metric, logits, target)
            timer.log(f'loss: {total_loss / (timer.current + 1):.4f} acc: {metric.get_metric():.2%}',
                      ratio_percentage=None,
                      logger=logger)
            del loss
        return total_loss / timer.total

    def update_metric(self, metric, logits: torch.Tensor, target, output=None):
        metric(logits, target)
        if output:
            label_ids = logits.argmax(-1)
            return label_ids

    def compute_loss(self, criterion, logits, target, batch):
        loss = criterion(logits, target)
        return loss

    def feed_batch(self, batch) -> torch.LongTensor:
        logits = self.model(*[batch[key] for key in ['input_ids', 'attention_mask', 'token_type_ids']])
        return logits

    # noinspection PyMethodOverriding
    def evaluate_dataloader(self,
                            data: DataLoader,
                            criterion: Callable,
                            metric,
                            logger,
                            ratio_width=None,
                            filename=None,
                            output=None,
                            **kwargs):
        self.model.eval()
        timer = CountdownTimer(len(data))
        total_loss = 0
        metric.reset()
        num_samples = 0
        if output:
            output = open(output, 'w')
        for batch in data:
            logits = self.feed_batch(batch)
            target = batch['label_id']
            loss = self.compute_loss(criterion, logits, target, batch)
            total_loss += loss.item()
            label_ids = self.update_metric(metric, logits, target, output)
            if output:
                labels = [self.vocabs[self.config.label_key].idx_to_token[i] for i in label_ids.tolist()]
                for i, label in enumerate(labels):
                    # text_a text_b pred gold
                    columns = [batch[self.config.text_a_key][i]]
                    if self.config.text_b_key:
                        columns.append(batch[self.config.text_b_key][i])
                    columns.append(label)
                    columns.append(batch[self.config.label_key][i])
                    output.write('\t'.join(columns))
                    output.write('\n')
            num_samples += len(target)
            report = f'loss: {total_loss / (timer.current + 1):.4f} acc: {metric.get_metric():.2%}'
            if filename:
                report = f'{filename} {report} {num_samples / timer.elapsed:.0f} samples/sec'
            timer.log(report, ratio_percentage=None, logger=logger, ratio_width=ratio_width)
        if output:
            output.close()
        return total_loss / timer.total

    # noinspection PyMethodOverriding
    def build_model(self, transformer, training=True, **kwargs) -> torch.nn.Module:
        # config: PretrainedConfig = AutoConfig.from_pretrained(transformer)
        # config.num_labels = len(self.vocabs.label)
        # config.hidden_dropout_prob = self.config.hidden_dropout_prob
        transformer = self.build_transformer(training=training).transformer
        model = TransformerClassificationModel(transformer, len(self.vocabs.label))
        # truncated_normal_(model.classifier.weight, mean=0.02, std=0.05)
        return model

    # noinspection PyMethodOverriding
    def build_dataloader(self, data, batch_size, shuffle, device, text_a_key, text_b_key,
                         label_key,
                         logger: logging.Logger = None,
                         sorting=True,
                         **kwargs) -> DataLoader:
        if not batch_size:
            batch_size = self.config.batch_size
        dataset = self.build_dataset(data)
        dataset.append_transform(self.vocabs)
        if self.vocabs.mutable:
            if not any([text_a_key, text_b_key]):
                if len(dataset.headers) == 2:
                    self.config.text_a_key = dataset.headers[0]
                    self.config.label_key = dataset.headers[1]
                elif len(dataset.headers) >= 3:
                    self.config.text_a_key, self.config.text_b_key, self.config.label_key = dataset.headers[0], \
                                                                                            dataset.headers[1], \
                                                                                            dataset.headers[-1]
                else:
                    raise ValueError('Wrong dataset format')
                report = {'text_a_key', 'text_b_key', 'label_key'}
                report = dict((k, self.config[k]) for k in report)
                report = [f'{k}={v}' for k, v in report.items() if v]
                report = ', '.join(report)
                logger.info(f'Guess [bold][blue]{report}[/blue][/bold] according to the headers of training dataset: '
                            f'[blue]{dataset}[/blue]')
            self.build_vocabs(dataset, logger)
            dataset.purge_cache()
        # if self.config.transform:
        #     dataset.append_transform(self.config.transform)
        dataset.append_transform(TransformerTextTokenizer(tokenizer=self.transformer_tokenizer,
                                                          text_a_key=self.config.text_a_key,
                                                          text_b_key=self.config.text_b_key,
                                                          max_seq_length=self.config.max_seq_length,
                                                          truncate_long_sequences=self.config.truncate_long_sequences,
                                                          output_key=''))
        batch_sampler = None
        if sorting and not isdebugging():
            if dataset.cache and len(dataset) > 1000:
                timer = CountdownTimer(len(dataset))
                lens = []
                for idx, sample in enumerate(dataset):
                    lens.append(len(sample['input_ids']))
                    timer.log('Pre-processing and caching dataset [blink][yellow]...[/yellow][/blink]',
                              ratio_percentage=None)
            else:
                lens = [len(sample['input_ids']) for sample in dataset]
            batch_sampler = SortingSampler(lens, batch_size=batch_size, shuffle=shuffle,
                                           batch_max_tokens=self.config.batch_max_tokens)
        return PadSequenceDataLoader(dataset, batch_size, shuffle, batch_sampler=batch_sampler, device=device)

    def build_dataset(self, data) -> TransformableDataset:
        if isinstance(data, str):
            dataset = TableDataset(data, cache=True)
        elif isinstance(data, TableDataset):
            dataset = data
        elif isinstance(data, list):
            dataset = TableDataset(data)
        else:
            raise ValueError(f'Unsupported data {data}')
        return dataset

    def predict(self, data: Union[str, List[str]], batch_size: int = None, **kwargs):
        if not data:
            return []
        flat = isinstance(data, str) or isinstance(data, tuple)
        if flat:
            data = [data]
        samples = []
        for idx, d in enumerate(data):
            sample = {IDX: idx}
            if self.config.text_b_key:
                sample[self.config.text_a_key] = d[0]
                sample[self.config.text_b_key] = d[1]
            else:
                sample[self.config.text_a_key] = d
            samples.append(sample)
        dataloader = self.build_dataloader(samples,
                                           sorting=False,
                                           **merge_dict(self.config,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        device=self.device,
                                                        overwrite=True)
                                           )
        labels = [None] * len(data)
        vocab = self.vocabs.label
        for batch in dataloader:
            logits = self.feed_batch(batch)
            pred = logits.argmax(-1)
            pred = pred.tolist()
            for idx, tag in zip(batch[IDX], pred):
                labels[idx] = vocab.idx_to_token[tag]
        if flat:
            return labels[0]
        return labels

    def fit(self, trn_data, dev_data, save_dir,
            text_a_key=None,
            text_b_key=None,
            label_key=None,
            transformer=None,
            max_seq_length=512,
            truncate_long_sequences=True,
            # hidden_dropout_prob=0.0,
            lr=5e-5,
            transformer_lr=None,
            adam_epsilon=1e-6,
            weight_decay=0,
            warmup_steps=0.1,
            batch_size=32,
            batch_max_tokens=None,
            epochs=3,
            logger=None,
            # transform=None,
            devices: Union[float, int, List[int]] = None,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_vocabs(self, trn, logger, **kwargs):
        self.vocabs.label = Vocab(pad_token=None, unk_token=None)
        for each in trn:
            pass
        self.vocabs.lock()
        self.vocabs.summary(logger)
