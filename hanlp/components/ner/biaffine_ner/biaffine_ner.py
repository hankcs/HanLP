# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-09 18:13
import logging
from typing import Union, List, Callable, Dict, Any

from hanlp_common.constant import IDX
from hanlp.common.structure import History
from hanlp.components.ner.biaffine_ner.biaffine_ner_model import BiaffineNamedEntityRecognitionModel
from hanlp.datasets.ner.json_ner import JsonNERDataset, unpack_ner
from hanlp.layers.transformers.utils import build_optimizer_scheduler_with_transformer
import torch
from torch.utils.data import DataLoader
from hanlp.common.dataset import PadSequenceDataLoader
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import FieldLength, TransformList
from hanlp.common.vocab import Vocab
from hanlp.layers.embeddings.embedding import Embedding
from hanlp.metrics.f1 import F1
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.util import merge_locals_kwargs, reorder


class BiaffineNamedEntityRecognizer(TorchComponent):

    def __init__(self, **kwargs) -> None:
        """An implementation of Named Entity Recognition as Dependency Parsing (:cite:`yu-etal-2020-named`). It treats
        every possible span as a candidate of entity and predicts its entity label. Non-entity spans are assigned NULL
        label to be excluded. The label prediction is done with a biaffine layer (:cite:`dozat:17a`). As it makes no
        assumption about the spans, it naturally supports flat NER and nested NER.

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)
        self.model: BiaffineNamedEntityRecognitionModel = None

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
        if self.use_transformer:
            num_training_steps = len(trn) * epochs // self.config.get('gradient_accumulation', 1)
            optimizer, scheduler = build_optimizer_scheduler_with_transformer(self.model,
                                                                              self._get_transformer(),
                                                                              lr, transformer_lr,
                                                                              num_training_steps, warmup_steps,
                                                                              weight_decay, adam_epsilon)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), self.config.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='max',
                factor=0.5,
                patience=2,
                verbose=True,
            )
        return optimizer, scheduler

    @property
    def use_transformer(self):
        return 'token' not in self.vocabs

    def _get_transformer(self):
        return getattr(self.model_.embed, 'transformer', None)

    def build_criterion(self, **kwargs):
        pass

    # noinspection PyProtectedMember
    def build_metric(self, **kwargs) -> F1:
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
                              gradient_accumulation=1,
                              **kwargs):
        best_epoch, best_metric = 0, -1
        optimizer, scheduler = optimizer
        history = History()
        timer = CountdownTimer(epochs)
        ratio_width = len(f'{len(trn)}/{len(trn)}')
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, history=history,
                                gradient_accumulation=gradient_accumulation,
                                linear_scheduler=scheduler if self._get_transformer() else None)
            if dev:
                self.evaluate_dataloader(dev, criterion, metric, logger, ratio_width=ratio_width)
            report = f'{timer.elapsed_human}/{timer.total_time_human}'
            dev_score = metric.score
            if not self._get_transformer():
                scheduler.step(dev_score)
            if dev_score > best_metric:
                self.save_weights(save_dir)
                best_metric = dev_score
                report += ' [red]saved[/red]'
            timer.log(report, ratio_percentage=False, newline=True, ratio=False)
        return best_metric

    def fit_dataloader(self,
                       trn: DataLoader,
                       criterion,
                       optimizer,
                       metric,
                       logger: logging.Logger,
                       linear_scheduler=None,
                       history: History = None,
                       gradient_accumulation=1,
                       **kwargs):
        self.model.train()
        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation=gradient_accumulation))
        total_loss = 0
        self.reset_metrics(metric)
        for batch in trn:
            optimizer.zero_grad()
            output_dict = self.feed_batch(batch)
            self.update_metrics(batch, output_dict, metric)
            loss = output_dict['loss']
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if history.step(gradient_accumulation):
                if self.config.grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
                optimizer.step()
                if linear_scheduler:
                    linear_scheduler.step()
                timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                          logger=logger)
            del loss
        return total_loss / timer.total

    # noinspection PyMethodOverriding
    @torch.no_grad()
    def evaluate_dataloader(self,
                            data: DataLoader,
                            criterion: Callable,
                            metric,
                            logger,
                            ratio_width=None,
                            output=False,
                            **kwargs):
        self.model.eval()
        self.reset_metrics(metric)
        timer = CountdownTimer(len(data))
        total_loss = 0
        if output:
            fp = open(output, 'w')
        for batch in data:
            output_dict = self.feed_batch(batch)
            if output:
                for sent, pred, gold in zip(batch['token'], output_dict['prediction'], batch['ner']):
                    fp.write('Tokens\t' + ' '.join(sent) + '\n')
                    fp.write('Pred\t' + '\t'.join(
                        ['[' + ' '.join(sent[x:y + 1]) + f']/{label}' for x, y, label in pred]) + '\n')
                    fp.write('Gold\t' + '\t'.join(
                        ['[' + ' '.join(sent[x:y + 1]) + f']/{label}' for x, y, label in gold]) + '\n')
                    fp.write('\n')
            self.update_metrics(batch, output_dict, metric)
            loss = output_dict['loss']
            total_loss += loss.item()
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                      logger=logger,
                      ratio_width=ratio_width)
            del loss
        if output:
            fp.close()
        return total_loss / timer.total, metric

    def build_model(self,
                    training=True,
                    **kwargs) -> torch.nn.Module:
        # noinspection PyTypeChecker
        # embed: torch.nn.Embedding = self.config.embed.module(vocabs=self.vocabs)[0].embed
        model = BiaffineNamedEntityRecognitionModel(self.config,
                                                    self.config.embed.module(vocabs=self.vocabs),
                                                    self.config.context_layer,
                                                    len(self.vocabs.label))
        return model

    # noinspection PyMethodOverriding
    def build_dataloader(self, data, batch_size, shuffle, device, logger: logging.Logger = None, vocabs=None,
                         sampler_builder=None,
                         gradient_accumulation=1,
                         **kwargs) -> DataLoader:
        if vocabs is None:
            vocabs = self.vocabs
        transform = TransformList(unpack_ner, FieldLength('token'))
        if isinstance(self.config.embed, Embedding):
            transform.append(self.config.embed.transform(vocabs=vocabs))
        transform.append(self.vocabs)
        dataset = self.build_dataset(data, vocabs, transform)
        if vocabs.mutable:
            self.build_vocabs(dataset, logger, vocabs)
        if 'token' in vocabs:
            lens = [x['token'] for x in dataset]
        else:
            lens = [len(x['token_input_ids']) for x in dataset]
        if sampler_builder:
            sampler = sampler_builder.build(lens, shuffle, gradient_accumulation)
        else:
            sampler = None
        return PadSequenceDataLoader(batch_sampler=sampler,
                                     device=device,
                                     dataset=dataset)

    def build_dataset(self, data, vocabs, transform):
        dataset = JsonNERDataset(data, transform=transform,
                                 doc_level_offset=self.config.get('doc_level_offset', True),
                                 tagset=self.config.get('tagset', None))
        dataset.append_transform(vocabs)
        if isinstance(data, str):
            dataset.purge_cache()  # Enable cache
        return dataset

    def predict(self, data: Union[List[str], List[List[str]]], batch_size: int = None, ret_tokens=True, **kwargs):
        if not data:
            return []
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        dataloader = self.build_dataloader([{'token': x} for x in data], batch_size, False, self.device)
        predictions = []
        orders = []
        for batch in dataloader:
            output_dict = self.feed_batch(batch)
            token = batch['token']
            prediction = output_dict['prediction']
            self.prediction_to_result(token, prediction, predictions, ret_tokens)
            orders.extend(batch[IDX])
        predictions = reorder(predictions, orders)
        if flat:
            return predictions[0]
        return predictions

    @staticmethod
    def prediction_to_result(token, prediction, predictions: List, ret_tokens: Union[bool, str]):
        for tokens, ner in zip(token, prediction):
            prediction_per_sent = []
            for i, (b, e, l) in enumerate(ner):
                if ret_tokens is not None:
                    entity = tokens[b: e + 1]
                    if isinstance(ret_tokens, str):
                        entity = ret_tokens.join(entity)
                    prediction_per_sent.append((entity, l, b, e + 1))
                else:
                    prediction_per_sent.append((b, e + 1, l))
            predictions.append(prediction_per_sent)

    @staticmethod
    def input_is_flat(data):
        return isinstance(data[0], str)

    # noinspection PyMethodOverriding
    def fit(self,
            trn_data,
            dev_data,
            save_dir,
            embed: Embedding,
            context_layer,
            sampler='sorting',
            n_buckets=32,
            batch_size=50,
            lexical_dropout=0.5,
            ffnn_size=150,
            is_flat_ner=True,
            doc_level_offset=True,
            lr=1e-3,
            transformer_lr=1e-5,
            adam_epsilon=1e-6,
            weight_decay=0.01,
            warmup_steps=0.1,
            grad_norm=5.0,
            epochs=50,
            loss_reduction='sum',
            gradient_accumulation=1,
            ret_tokens=True,
            tagset=None,
            sampler_builder=None,
            devices=None,
            logger=None,
            seed=None,
            **kwargs
            ):
        """

        Args:
            trn_data: Path to training set.
            dev_data: Path to dev set.
            save_dir: The directory to save trained component.
            embed: Embeddings to use.
            context_layer: A contextualization layer (transformer or RNN).
            sampler: Sampler to use.
            n_buckets: Number of buckets to use in KMeans sampler.
            batch_size: The number of samples in a batch.
            lexical_dropout: Dropout applied to hidden states of context layer.
            ffnn_size: Feedforward size for MLPs extracting the head/tail representations.
            is_flat_ner: ``True`` for flat NER, otherwise nested NER.
            doc_level_offset: ``True`` to indicate the offsets in ``jsonlines`` are of document level.
            lr: Learning rate for decoder.
            transformer_lr: Learning rate for encoder.
            adam_epsilon: The epsilon to use in Adam.
            weight_decay: The weight decay to use.
            warmup_steps: The number of warmup steps.
            grad_norm: Gradient norm for clipping.
            epochs: The number of epochs to train.
            loss_reduction: The loss reduction used in aggregating losses.
            gradient_accumulation: Number of mini-batches per update step.
            ret_tokens: A delimiter between tokens in entities so that the surface form of an entity can be rebuilt.
            tagset: Optional tagset to prune entities outside of this tagset from datasets.
            sampler_builder: The builder to build sampler, which will override batch_size.
            devices: Devices this component will live on.
            logger: Any :class:`logging.Logger` instance.
            seed: Random seed to reproduce this training.
            **kwargs: Not used.

        Returns:
            The best metrics on training set.
        """
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_vocabs(self, dataset, logger, vocabs, lock=True, label_vocab_name='label', **kwargs):
        vocabs[label_vocab_name] = label_vocab = Vocab(pad_token=None, unk_token=None)
        # Use null to indicate no relationship
        label_vocab.add('<null>')
        timer = CountdownTimer(len(dataset))
        for each in dataset:
            timer.log('Building NER vocab [blink][yellow]...[/yellow][/blink]')
        label_vocab.set_unk_as_safe_unk()
        if lock:
            vocabs.lock()
            vocabs.summary(logger)

    def reset_metrics(self, metrics):
        metrics.reset()

    def report_metrics(self, loss, metrics):
        return f'loss: {loss:.4f} {metrics}'

    def feed_batch(self, batch) -> Dict[str, Any]:
        output_dict = self.model(batch)
        output_dict['prediction'] = self.get_pred_ner(batch['token'], output_dict['candidate_ner_scores'])
        return output_dict

    def update_metrics(self, batch: dict, prediction: Union[Dict, List], metrics):
        if isinstance(prediction, dict):
            prediction = prediction['prediction']
        assert len(prediction) == len(batch['ner'])
        for pred, gold in zip(prediction, batch['ner']):
            metrics(set(pred), set(gold))

    def get_pred_ner(self, sentences, span_scores):
        is_flat_ner = self.config.is_flat_ner
        candidates = []
        for sid, sent in enumerate(sentences):
            for s in range(len(sent)):
                for e in range(s, len(sent)):
                    candidates.append((sid, s, e))

        top_spans = [[] for _ in range(len(sentences))]
        span_scores_cpu = span_scores.tolist()
        for i, type in enumerate(torch.argmax(span_scores, dim=-1).tolist()):
            if type > 0:
                sid, s, e = candidates[i]
                top_spans[sid].append((s, e, type, span_scores_cpu[i][type]))

        top_spans = [sorted(top_span, reverse=True, key=lambda x: x[3]) for top_span in top_spans]
        sent_pred_mentions = [[] for _ in range(len(sentences))]
        for sid, top_span in enumerate(top_spans):
            for ns, ne, t, _ in top_span:
                for ts, te, _ in sent_pred_mentions[sid]:
                    if ns < ts <= ne < te or ts < ns <= te < ne:
                        # for both nested and flat ner no clash is allowed
                        break
                    if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                        # for flat ner nested mentions are not allowed
                        break
                else:
                    sent_pred_mentions[sid].append((ns, ne, t))
        pred_mentions = set((sid, s, e, t) for sid, spr in enumerate(sent_pred_mentions) for s, e, t in spr)
        prediction = [[] for _ in range(len(sentences))]
        idx_to_label = self.vocabs['label'].idx_to_token
        for sid, s, e, t in sorted(pred_mentions):
            prediction[sid].append((s, e, idx_to_label[t]))
        return prediction
