# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-20 17:03
import logging
from typing import Union, List
import torch
from torch.utils.data import DataLoader
from hanlp.common.structure import History
from hanlp.layers.transformers.pt_imports import AutoConfig_, AutoTokenizer_
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from hanlp.common.dataset import SortingSamplerBuilder, PadSequenceDataLoader
from hanlp.common.torch_component import TorchComponent
from hanlp.datasets.sts.stsb import SemanticTextualSimilarityDataset
from hanlp.layers.transformers.utils import build_optimizer_scheduler_with_transformer
from hanlp.metrics.spearman_correlation import SpearmanCorrelation
from hanlp.transform.transformer_tokenizer import TransformerTextTokenizer
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.util import merge_locals_kwargs, reorder
from hanlp_common.constant import IDX


class TransformerSemanticTextualSimilarity(TorchComponent):

    def __init__(self, **kwargs) -> None:
        """
        A simple Semantic Textual Similarity (STS) baseline which fine-tunes a transformer with a regression layer on
        top of it.

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)
        self._tokenizer = None

    # noinspection PyMethodOverriding
    def build_dataloader(self, data, batch_size, sent_a_col=None,
                         sent_b_col=None,
                         similarity_col=None,
                         delimiter='auto',
                         gradient_accumulation=1,
                         sampler_builder=None,
                         shuffle=False, device=None, logger: logging.Logger = None,
                         split=None,
                         **kwargs) -> DataLoader:
        dataset = SemanticTextualSimilarityDataset(data,
                                                   sent_a_col,
                                                   sent_b_col,
                                                   similarity_col,
                                                   delimiter=delimiter,
                                                   transform=self._tokenizer,
                                                   cache=isinstance(data, str))
        if split == 'trn':
            scores = [x['similarity'] for x in dataset]
            self.config.max_score = max(scores)
            self.config.min_score = min(scores)
        if not sampler_builder:
            sampler_builder = SortingSamplerBuilder(batch_size=batch_size)
        lens = [len(x['input_ids']) for x in dataset]
        return PadSequenceDataLoader(dataset, batch_sampler=sampler_builder.build(lens, shuffle, gradient_accumulation),
                                     device=device,
                                     pad={'similarity': 0.0, 'input_ids': self._tokenizer.tokenizer.pad_token_id})

    def build_optimizer(self, trn, epochs, gradient_accumulation=1, lr=1e-3, transformer_lr=5e-5, adam_epsilon=1e-8,
                        weight_decay=0.0, warmup_steps=0.1, **kwargs):
        num_training_steps = len(trn) * epochs // gradient_accumulation
        optimizer, scheduler = build_optimizer_scheduler_with_transformer(self.model,
                                                                          self.model.base_model,
                                                                          lr, transformer_lr,
                                                                          num_training_steps, warmup_steps,
                                                                          weight_decay, adam_epsilon)
        return optimizer, scheduler

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs):
        return SpearmanCorrelation()

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, gradient_accumulation=1, **kwargs):
        best_epoch, best_metric = 0, -1
        timer = CountdownTimer(epochs)
        history = History()
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, ratio_width=ratio_width,
                                gradient_accumulation=gradient_accumulation, history=history, save_dir=save_dir)
            report = f'{timer.elapsed_human}/{timer.total_time_human}'
            self.evaluate_dataloader(dev, logger, ratio_width=ratio_width, save_dir=save_dir, metric=metric)
            if metric > best_metric:
                self.save_weights(save_dir)
                best_metric = float(metric)
                best_epoch = epoch
                report += ' [red]saved[/red]'
            timer.log(report, ratio_percentage=False, newline=True, ratio=False)
        if best_epoch and best_epoch != epochs:
            logger.info(f'Restored the best model with {best_metric} saved {epochs - best_epoch} epochs ago')
            self.load_weights(save_dir)

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric: SpearmanCorrelation, logger: logging.Logger,
                       history=None, gradient_accumulation=1, **kwargs):
        self.model.train()
        optimizer, scheduler = optimizer
        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation=gradient_accumulation))
        total_loss = 0
        metric.reset()
        for batch in trn:
            output = self.feed_batch(batch)
            prediction = self.decode(output)
            metric(prediction, batch['similarity'])
            loss = output['loss']
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if history.step(gradient_accumulation):
                if self.config.grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                          logger=logger)
            del loss
        return total_loss / timer.total

    @torch.no_grad()
    def evaluate_dataloader(self, data: DataLoader, logger: logging.Logger, metric=None, output=False, **kwargs):
        self.model.eval()
        timer = CountdownTimer(len(data))
        total_loss = 0
        metric.reset()
        if output:
            predictions = []
            orders = []
            samples = []
        for batch in data:
            output_dict = self.feed_batch(batch)
            prediction = self.decode(output_dict)
            metric(prediction, batch['similarity'])
            if output:
                predictions.extend(prediction.tolist())
                orders.extend(batch[IDX])
                samples.extend(list(zip(batch['sent_a'], batch['sent_b'])))
            loss = output_dict['loss']
            total_loss += loss.item()
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                      logger=logger)
            del loss
        if output:
            predictions = reorder(predictions, orders)
            samples = reorder(samples, orders)
            with open(output, 'w') as out:
                for s, p in zip(samples, predictions):
                    out.write('\t'.join(s + (str(p),)))
                    out.write('\n')
        return total_loss / timer.total

    # noinspection PyMethodOverriding
    def build_model(self, transformer, training=True, **kwargs) -> torch.nn.Module:
        config = AutoConfig_.from_pretrained(transformer, num_labels=1)
        if training:
            model = AutoModelForSequenceClassification.from_pretrained(transformer, config=config)
        else:
            model = AutoModelForSequenceClassification.from_config(config)
        return model

    def predict(self, data: Union[List[str], List[List[str]]], batch_size: int = None, **kwargs) -> Union[
        float, List[float]]:
        """ Predict the similarity between sentence pairs.

        Args:
            data: Sentence pairs.
            batch_size: The number of samples in a batch.
            **kwargs: Not used.

        Returns:
            Similarities between sentences.
        """
        if not data:
            return []
        flat = isinstance(data[0], str)
        if flat:
            data = [data]
        dataloader = self.build_dataloader([{'sent_a': x[0], 'sent_b': x[1]} for x in data],
                                           batch_size=batch_size or self.config.batch_size,
                                           device=self.device)
        orders = []
        predictions = []
        for batch in dataloader:
            output_dict = self.feed_batch(batch)
            prediction = self.decode(output_dict)
            predictions.extend(prediction.tolist())
            orders.extend(batch[IDX])
        predictions = reorder(predictions, orders)
        if flat:
            return predictions[0]
        return predictions

    # noinspection PyMethodOverriding
    def fit(self, trn_data, dev_data, save_dir,
            transformer,
            sent_a_col,
            sent_b_col,
            similarity_col,
            delimiter='auto',
            batch_size=32,
            max_seq_len=128,
            epochs=3,
            lr=1e-3,
            transformer_lr=5e-5,
            adam_epsilon=1e-8,
            weight_decay=0.0,
            warmup_steps=0.1,
            gradient_accumulation=1,
            grad_norm=1.0,
            sampler_builder=None,
            devices=None,
            logger=None,
            seed=None,
            finetune: Union[bool, str] = False, eval_trn=True, _device_placeholder=False, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def on_config_ready(self, transformer, max_seq_len, **kwargs):
        super().on_config_ready(**kwargs)
        self._tokenizer = TransformerTextTokenizer(AutoTokenizer_.from_pretrained(transformer),
                                                   text_a_key='sent_a',
                                                   text_b_key='sent_b',
                                                   output_key='',
                                                   max_seq_length=max_seq_len)

    def feed_batch(self, batch) -> SequenceClassifierOutput:
        return self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                          token_type_ids=batch['token_type_ids'], labels=batch.get('similarity', None))

    def decode(self, output: SequenceClassifierOutput):
        return output.logits.squeeze(-1).detach().clip(self.config.min_score, self.config.max_score)

    def report_metrics(self, loss, metric):
        return f'loss: {loss:.4f} {metric}'
