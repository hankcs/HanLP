# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-09 18:13
import logging
from bisect import bisect
from typing import Union, List, Callable, Tuple, Dict, Any

from hanlp_common.constant import IDX
from hanlp.layers.transformers.utils import build_optimizer_scheduler_with_transformer
import torch
from torch.utils.data import DataLoader
from hanlp.common.dataset import PadSequenceDataLoader, SortingSampler
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import FieldLength
from hanlp.common.vocab import Vocab
from hanlp.components.srl.span_rank.inference_utils import srl_decode
from hanlp.components.srl.span_rank.span_ranking_srl_model import SpanRankingSRLModel
from hanlp.components.srl.span_rank.srl_eval_utils import compute_srl_f1
from hanlp.datasets.srl.conll2012 import CoNLL2012SRLDataset, filter_v_args, unpack_srl, \
    group_pa_by_p
from hanlp.layers.embeddings.embedding import Embedding
from hanlp.metrics.f1 import F1
from hanlp_common.visualization import markdown_table
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.util import merge_locals_kwargs, reorder


class SpanRankingSemanticRoleLabeler(TorchComponent):
    def __init__(self, **kwargs) -> None:
        """An implementation of "Jointly Predicting Predicates and Arguments in Neural Semantic Role Labeling"
        (:cite:`he-etal-2018-jointly`). It generates candidates triples of (predicate, arg_start, arg_end) and rank them.

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)
        self.model: SpanRankingSRLModel = None

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
        transformer = self._get_transformer()
        if transformer:
            num_training_steps = len(trn) * epochs // self.config.get('gradient_accumulation', 1)
            optimizer, scheduler = build_optimizer_scheduler_with_transformer(self.model,
                                                                              transformer,
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

    def _get_transformer(self):
        return getattr(self.model_.embed, 'transformer', None)

    def build_criterion(self, **kwargs):
        pass

    # noinspection PyProtectedMember
    def build_metric(self, **kwargs) -> Tuple[F1, F1]:
        predicate_f1 = F1()
        end_to_end_f1 = F1()
        return predicate_f1, end_to_end_f1

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
        predicate, end_to_end = metric
        optimizer, scheduler = optimizer
        timer = CountdownTimer(epochs)
        ratio_width = len(f'{len(trn)}/{len(trn)}')
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger,
                                linear_scheduler=scheduler if self._get_transformer() else None)
            if dev:
                self.evaluate_dataloader(dev, criterion, metric, logger, ratio_width=ratio_width)
            report = f'{timer.elapsed_human}/{timer.total_time_human}'
            dev_score = end_to_end.score
            if not self._get_transformer():
                scheduler.step(dev_score)
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
                       gradient_accumulation=1,
                       **kwargs):
        self.model.train()
        timer = CountdownTimer(len(trn) // gradient_accumulation)
        total_loss = 0
        self.reset_metrics(metric)
        for idx, batch in enumerate(trn):
            output_dict = self.feed_batch(batch)
            self.update_metrics(batch, output_dict, metric)
            loss = output_dict['loss']
            loss = loss.sum()  # For data parallel
            loss.backward()
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            if self.config.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
            if (idx + 1) % gradient_accumulation == 0:
                self._step(optimizer, linear_scheduler)
                timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                          logger=logger)
            total_loss += loss.item()
            del loss
        if len(trn) % gradient_accumulation:
            self._step(optimizer, linear_scheduler)
        return total_loss / timer.total

    def _step(self, optimizer, linear_scheduler):
        optimizer.step()
        optimizer.zero_grad()
        if linear_scheduler:
            linear_scheduler.step()

    # noinspection PyMethodOverriding
    @torch.no_grad()
    def evaluate_dataloader(self,
                            data: DataLoader,
                            criterion: Callable,
                            metric,
                            logger,
                            ratio_width=None,
                            output=False,
                            official=False,
                            confusion_matrix=False,
                            **kwargs):
        self.model.eval()
        self.reset_metrics(metric)
        timer = CountdownTimer(len(data))
        total_loss = 0
        if official:
            sentences = []
            gold = []
            pred = []
        for batch in data:
            output_dict = self.feed_batch(batch)
            if official:
                sentences += batch['token']
                gold += batch['srl']
                pred += output_dict['prediction']
            self.update_metrics(batch, output_dict, metric)
            loss = output_dict['loss']
            total_loss += loss.item()
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                      logger=logger,
                      ratio_width=ratio_width)
            del loss
        if official:
            scores = compute_srl_f1(sentences, gold, pred)
            if logger:
                if confusion_matrix:
                    labels = sorted(set(y for x in scores.label_confusions.keys() for y in x))
                    headings = ['GOLD↓PRED→'] + labels
                    matrix = []
                    for i, gold in enumerate(labels):
                        row = [gold]
                        matrix.append(row)
                        for j, pred in enumerate(labels):
                            row.append(scores.label_confusions.get((gold, pred), 0))
                    matrix = markdown_table(headings, matrix)
                    logger.info(f'{"Confusion Matrix": ^{len(matrix.splitlines()[0])}}')
                    logger.info(matrix)
                headings = ['Settings', 'Precision', 'Recall', 'F1']
                data = []
                for h, (p, r, f) in zip(['Unlabeled', 'Labeled', 'Official'], [
                    [scores.unlabeled_precision, scores.unlabeled_recall, scores.unlabeled_f1],
                    [scores.precision, scores.recall, scores.f1],
                    [scores.conll_precision, scores.conll_recall, scores.conll_f1],
                ]):
                    data.append([h] + [f'{x:.2%}' for x in [p, r, f]])
                table = markdown_table(headings, data)
                logger.info(f'{"Scores": ^{len(table.splitlines()[0])}}')
                logger.info(table)
        else:
            scores = metric
        return total_loss / timer.total, scores

    def build_model(self,
                    training=True,
                    **kwargs) -> torch.nn.Module:
        # noinspection PyTypeChecker
        # embed: torch.nn.Embedding = self.config.embed.module(vocabs=self.vocabs)[0].embed
        model = SpanRankingSRLModel(self.config,
                                    self.config.embed.module(vocabs=self.vocabs, training=training),
                                    self.config.context_layer,
                                    len(self.vocabs.srl_label))
        return model

    # noinspection PyMethodOverriding
    def build_dataloader(self, data, batch_size, shuffle, device, logger: logging.Logger,
                         generate_idx=False, **kwargs) -> DataLoader:
        batch_max_tokens = self.config.batch_max_tokens
        gradient_accumulation = self.config.get('gradient_accumulation', 1)
        if batch_size:
            batch_size //= gradient_accumulation
        if batch_max_tokens:
            batch_max_tokens //= gradient_accumulation
        dataset = self.build_dataset(data, generate_idx, logger)

        sampler = SortingSampler([x['token_length'] for x in dataset],
                                 batch_size=batch_size,
                                 batch_max_tokens=batch_max_tokens,
                                 shuffle=shuffle)
        return PadSequenceDataLoader(batch_sampler=sampler,
                                     device=device,
                                     dataset=dataset)

    def build_dataset(self, data, generate_idx, logger, transform=None):
        dataset = CoNLL2012SRLDataset(data, transform=[filter_v_args, unpack_srl, group_pa_by_p],
                                      doc_level_offset=self.config.doc_level_offset, generate_idx=generate_idx)
        if transform:
            dataset.append_transform(transform)
        if isinstance(self.config.get('embed', None), Embedding):
            transform = self.config.embed.transform(vocabs=self.vocabs)
            if transform:
                dataset.append_transform(transform)
        dataset.append_transform(self.vocabs)
        dataset.append_transform(FieldLength('token'))
        if isinstance(data, str):
            dataset.purge_cache()  # Enable cache
        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger)
        return dataset

    def predict(self, data: Union[str, List[str]], batch_size: int = None, fmt='dict', **kwargs):
        if not data:
            return []
        flat = self.input_is_flat(data)
        if flat:
            data = [data]
        samples = []
        for token in data:
            sample = dict()
            sample['token'] = token
            samples.append(sample)
        batch_size = batch_size or self.config.batch_size
        dataloader = self.build_dataloader(samples, batch_size, False, self.device, None, generate_idx=True)
        outputs = []
        order = []
        for batch in dataloader:
            output_dict = self.feed_batch(batch)
            outputs.extend(output_dict['prediction'])
            order.extend(batch[IDX])
        outputs = reorder(outputs, order)
        if fmt == 'list':
            outputs = self.format_dict_to_results(data, outputs)
        if flat:
            return outputs[0]
        return outputs

    @staticmethod
    def format_dict_to_results(data, outputs, exclusive_offset=False, with_predicate=False, with_argument=False,
                               label_first=False):
        results = []
        for i in range(len(outputs)):
            tokens = data[i]
            output = []
            for p, a in outputs[i].items():
                # a: [(0, 0, 'ARG0')]
                if with_predicate:
                    a.insert(bisect([x[0] for x in a], p), (p, p, 'PRED'))
                if with_argument is not False:
                    a = [x + (tokens[x[0]:x[1] + 1],) for x in a]
                    if isinstance(with_argument, str):
                        a = [x[:-1] + (with_argument.join(x[-1]),) for x in a]
                if exclusive_offset:
                    a = [(x[0], x[1] + 1) + x[2:] for x in a]
                if label_first:
                    a = [tuple(reversed(x[2:])) + x[:2] for x in a]
                output.append(a)
            results.append(output)
        return results

    def input_is_flat(self, data):
        return isinstance(data[0], str)

    # noinspection PyMethodOverriding
    def fit(self,
            trn_data,
            dev_data,
            save_dir,
            embed,
            context_layer,
            batch_size=40,
            batch_max_tokens=700,
            lexical_dropout=0.5,
            dropout=0.2,
            span_width_feature_size=20,
            ffnn_size=150,
            ffnn_depth=2,
            argument_ratio=0.8,
            predicate_ratio=0.4,
            max_arg_width=30,
            mlp_label_size=100,
            enforce_srl_constraint=False,
            use_gold_predicates=False,
            doc_level_offset=True,
            use_biaffine=False,
            lr=1e-3,
            transformer_lr=1e-5,
            adam_epsilon=1e-6,
            weight_decay=0.01,
            warmup_steps=0.1,
            grad_norm=5.0,
            gradient_accumulation=1,
            loss_reduction='sum',
            devices=None,
            logger=None,
            seed=None,
            **kwargs
            ):

        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_vocabs(self, dataset, logger, **kwargs):
        self.vocabs.srl_label = Vocab(pad_token=None, unk_token=None)
        # Use null to indicate no relationship
        self.vocabs.srl_label.add('<null>')
        timer = CountdownTimer(len(dataset))
        max_seq_len = 0
        for each in dataset:
            max_seq_len = max(max_seq_len, len(each['token_input_ids']))
            timer.log(f'Building vocabs (max sequence length {max_seq_len}) [blink][yellow]...[/yellow][/blink]')
            pass
        timer.stop()
        timer.erase()
        self.vocabs['srl_label'].set_unk_as_safe_unk()
        self.vocabs.lock()
        self.vocabs.summary(logger)

    def reset_metrics(self, metrics):
        for each in metrics:
            each.reset()

    def report_metrics(self, loss, metrics):
        predicate, end_to_end = metrics
        return f'loss: {loss:.4f} predicate: {predicate.score:.2%} end_to_end: {end_to_end.score:.2%}'

    def feed_batch(self, batch) -> Dict[str, Any]:
        output_dict = self.model(batch)
        prediction = self.decode_output(output_dict, batch, self.model.training)
        output_dict['prediction'] = prediction
        return output_dict

    def decode_output(self, output_dict, batch, training=False):
        idx_to_label = self.vocabs['srl_label'].idx_to_token
        if training:
            # Use fast decoding during training,
            prediction = []
            top_predicate_indices = output_dict['predicates'].tolist()
            top_spans = torch.stack([output_dict['arg_starts'], output_dict['arg_ends']], dim=-1).tolist()
            srl_mask = output_dict['srl_mask'].tolist()
            for n, (pal, predicate_indices, argument_spans) in enumerate(
                    zip(output_dict['srl_scores'].argmax(-1).tolist(), top_predicate_indices, top_spans)):
                srl_per_sentence = {}
                for p, (al, predicate_index) in enumerate(zip(pal, predicate_indices)):
                    for a, (l, argument_span) in enumerate(zip(al, argument_spans)):
                        if l and srl_mask[n][p][a]:
                            args = srl_per_sentence.get(p, None)
                            if args is None:
                                args = srl_per_sentence[p] = []
                            args.append((*argument_span, idx_to_label[l]))
                prediction.append(srl_per_sentence)
        else:
            prediction = srl_decode(batch['token_length'], output_dict, idx_to_label, self.config)
        return prediction

    def update_metrics(self, batch: dict, output_dict: dict, metrics):
        def unpack(y: dict):
            return set((p, bel) for p, a in y.items() for bel in a)

        predicate, end_to_end = metrics
        for pred, gold in zip(output_dict['prediction'], batch['srl']):
            predicate(pred.keys(), gold.keys())
            end_to_end(unpack(pred), unpack(gold))
