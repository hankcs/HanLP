# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-05 20:28
import logging
from typing import Union, List, Callable, Dict, Any, Tuple

import torch
from hanlp.layers.transformers.utils import get_optimizers
from alnlp.metrics.conll_coref_scores import ConllCorefScores
from alnlp.metrics.mention_recall import MentionRecall
from alnlp.models.coref import CoreferenceResolver
from alnlp.modules.initializers import InitializerApplicator
from alnlp.modules.util import lengths_to_mask
from alnlp.training.optimizers import make_parameter_groups
from torch.utils.data import DataLoader
from hanlp.common.dataset import PadSequenceDataLoader
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import FieldLength
from hanlp.datasets.coref.conll12coref import CONLL12CorefDataset
from hanlp.layers.context_layer import LSTMContextualEncoder
from hanlp.layers.embeddings.embedding import Embedding
from hanlp.layers.feed_forward import FeedForward
from hanlp.utils.time_util import CountdownTimer
from hanlp.utils.torch_util import clip_grad_norm
from hanlp_common.util import merge_locals_kwargs


class CoreferenceResolverModel(CoreferenceResolver):
    # noinspection PyMethodOverriding
    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        batch['mask'] = mask = lengths_to_mask(batch['text_length'])
        return super().forward(batch, batch['spans'], batch.get('span_labels'), mask=mask)


class EndToEndCoreferenceResolver(TorchComponent):
    def build_optimizer(self,
                        trn,
                        epochs,
                        lr,
                        adam_epsilon,
                        weight_decay,
                        warmup_steps,
                        transformer_lr,
                        **kwargs):
        # noinspection PyProtectedMember
        transformer = getattr(self.model._text_field_embedder, 'transformer', None)
        if transformer:
            model = self.model
            num_training_steps = len(trn) * epochs // self.config.get('gradient_accumulation', 1)

            optimizer_grouped_parameters = make_parameter_groups(list(self.model.named_parameters()),
                                                                 [([".*transformer.*"], {"lr": transformer_lr})])
            optimizer, linear_scheduler = get_optimizers(model,
                                                         num_training_steps,
                                                         learning_rate=lr,
                                                         adam_epsilon=adam_epsilon,
                                                         weight_decay=weight_decay,
                                                         warmup_steps=warmup_steps,
                                                         optimizer_grouped_parameters=optimizer_grouped_parameters
                                                         )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), self.config.lr)
            linear_scheduler = None
        reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True,
        )
        return optimizer, reduce_lr_scheduler, linear_scheduler

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs) -> Tuple[MentionRecall, ConllCorefScores]:
        return self.model._mention_recall, self.model._conll_coref_scores

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
                              **kwargs):
        best_epoch, best_metric = 0, -1
        mention_recall, conll_coref_scores = self.build_metric()
        optimizer, reduce_lr_scheduler, linear_scheduler = optimizer
        timer = CountdownTimer(epochs)
        ratio_width = len(f'{len(trn)}/{len(trn)}')
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, linear_scheduler=linear_scheduler)
            if dev:
                self.evaluate_dataloader(dev, criterion, metric, logger, ratio_width=ratio_width)
            report = f'{timer.elapsed_human}/{timer.total_time_human}'
            dev_score = conll_coref_scores.get_metric()[-1]
            reduce_lr_scheduler.step(dev_score)
            if dev_score > best_metric:
                self.save_weights(save_dir)
                best_metric = dev_score
                report += ' [red]saved[/red]'
            timer.log(report, ratio_percentage=False, newline=True, ratio=False)

    def fit_dataloader(self,
                       trn: DataLoader,
                       criterion,
                       optimizer,
                       metric,
                       logger: logging.Logger,
                       linear_scheduler=None,
                       **kwargs):
        self.model.train()
        timer = CountdownTimer(len(trn))
        total_loss = 0
        self.reset_metrics()
        for batch in trn:
            optimizer.zero_grad()
            output_dict = self.feed_batch(batch)
            loss = output_dict['loss']
            loss.backward()
            if self.config.grad_norm:
                clip_grad_norm(self.model, self.config.grad_norm)
            optimizer.step()
            if linear_scheduler:
                linear_scheduler.step()
            total_loss += loss.item()
            timer.log(self.report_metrics(total_loss / (timer.current + 1)), ratio_percentage=None, logger=logger)
            del loss
        return total_loss / timer.total

    # noinspection PyMethodOverriding
    def evaluate_dataloader(self,
                            data: DataLoader,
                            criterion: Callable,
                            metric,
                            logger,
                            ratio_width=None,
                            output=False,
                            **kwargs):
        self.model.eval()
        self.reset_metrics()
        timer = CountdownTimer(len(data))
        total_loss = 0
        self.reset_metrics()
        for batch in data:
            output_dict = self.feed_batch(batch)
            loss = output_dict['loss']
            total_loss += loss.item()
            timer.log(self.report_metrics(total_loss / (timer.current + 1)), ratio_percentage=None, logger=logger,
                      ratio_width=ratio_width)
            del loss
        return total_loss / timer.total

    def build_model(self,
                    training=True,
                    **kwargs) -> torch.nn.Module:
        # noinspection PyTypeChecker
        model = CoreferenceResolverModel(
            self.config.embed.module(vocabs=self.vocabs, training=training),
            self.config.context_layer,
            self.config.mention_feedforward,
            self.config.antecedent_feedforward,
            self.config.feature_size,
            self.config.max_span_width,
            self.config.spans_per_word,
            self.config.max_antecedents,
            self.config.coarse_to_fine,
            self.config.inference_order,
            self.config.lexical_dropout,
            InitializerApplicator([
                [".*linear_layers.*weight", {"type": "xavier_normal"}],
                [".*scorer._module.weight", {"type": "xavier_normal"}],
                ["_distance_embedding.weight", {"type": "xavier_normal"}],
                ["_span_width_embedding.weight", {"type": "xavier_normal"}],
                ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
                ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
            ])
        )
        return model

    def build_dataloader(self,
                         data,
                         batch_size,
                         shuffle,
                         device,
                         logger: logging.Logger,
                         **kwargs) -> DataLoader:
        dataset = CONLL12CorefDataset(data, [FieldLength('text')])
        if isinstance(self.config.embed, Embedding):
            transform = self.config.embed.transform(vocabs=self.vocabs)
            if transform:
                dataset.append_transform(transform)
        dataset.append_transform(self.vocabs)
        if isinstance(data, str):
            dataset.purge_cache()  # Enable cache
        if self.vocabs.mutable:
            self.build_vocabs(dataset)
        return PadSequenceDataLoader(batch_size=batch_size,
                                     shuffle=shuffle,
                                     device=device,
                                     dataset=dataset,
                                     pad={'spans': 0, 'span_labels': -1})

    def predict(self, data: Union[str, List[str]], batch_size: int = None, **kwargs):
        pass

    # noinspection PyMethodOverriding
    def fit(self,
            trn_data,
            dev_data,
            save_dir,
            batch_size,
            embed: Embedding,
            mention_feedforward: FeedForward,
            antecedent_feedforward: FeedForward,
            feature_size: int,
            max_span_width: int,
            spans_per_word: float,
            max_antecedents: int,
            lr=1e-3,
            transformer_lr=1e-5,
            adam_epsilon=1e-6,
            weight_decay=0.01,
            warmup_steps=0.1,
            epochs=150,
            grad_norm=None,
            coarse_to_fine: bool = False,
            inference_order: int = 1,
            lexical_dropout: float = 0.2,
            context_layer: LSTMContextualEncoder = None,
            devices=None,
            logger=None,
            seed=None,
            **kwargs
            ):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def update_metric(self, metric, logits: torch.Tensor, target, output=None):
        metric(logits, target)
        if output:
            label_ids = logits.argmax(-1)
            return label_ids

    def compute_loss(self, criterion, logits, target, batch):
        loss = criterion(logits, target)
        return loss

    def feed_batch(self, batch) -> Dict[str, Any]:
        output_dict = self.model(batch)
        return output_dict

    def build_vocabs(self, dataset, **kwargs):
        if self.vocabs:
            for each in dataset:
                pass
            self.vocabs.lock()
            self.vocabs.summary()

    def reset_metrics(self):
        for each in self.build_metric():
            each.reset()

    def report_metrics(self, loss):
        mention_recall, conll_coref_scores = self.build_metric()
        return f'loss:{loss:.4f} mention_recall:{mention_recall.get_metric():.2%} f1:{conll_coref_scores.get_metric()[-1]:.2%}'
