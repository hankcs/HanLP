# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-20 19:55
import functools
import itertools
import logging
import os
from collections import defaultdict
from copy import copy
from typing import Union, List, Callable, Dict, Optional, Any, Iterable, Tuple, Set
from itertools import chain
import numpy as np
import torch
from alnlp.modules import util
from toposort import toposort
from torch.utils.data import DataLoader

from hanlp_common.constant import IDX, BOS, EOS
from hanlp.common.dataset import PadSequenceDataLoader, PrefetchDataLoader, CachedDataLoader
from hanlp_common.document import Document
from hanlp.common.structure import History
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import FieldLength, TransformList
from hanlp.components.mtl.tasks import Task
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbedding, ContextualWordEmbeddingModule
from hanlp.layers.embeddings.embedding import Embedding
from hanlp.layers.transformers.pt_imports import optimization
from hanlp.layers.transformers.utils import pick_tensor_for_each_token
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp_common.visualization import markdown_table
from hanlp.utils.time_util import CountdownTimer
from hanlp.utils.torch_util import clip_grad_norm
from hanlp_common.util import merge_locals_kwargs, topological_sort, reorder, prefix_match


class MultiTaskModel(torch.nn.Module):

    def __init__(self,
                 encoder: torch.nn.Module,
                 scalar_mixes: torch.nn.ModuleDict,
                 decoders: torch.nn.ModuleDict,
                 use_raw_hidden_states: dict) -> None:
        super().__init__()
        self.use_raw_hidden_states = use_raw_hidden_states
        self.encoder: ContextualWordEmbeddingModule = encoder
        self.scalar_mixes = scalar_mixes
        self.decoders = decoders


class MultiTaskDataLoader(DataLoader):

    def __init__(self, training=True, tau: float = 0.8, **dataloaders) -> None:
        # noinspection PyTypeChecker
        super().__init__(None)
        self.tau = tau
        self.training = training
        self.dataloaders: Dict[str, DataLoader] = dataloaders if dataloaders else {}
        # self.iterators = dict((k, iter(v)) for k, v in dataloaders.items())

    def __len__(self) -> int:
        if self.dataloaders:
            return sum(len(x) for x in self.dataloaders.values())
        return 0

    def __iter__(self):
        if self.training:
            sampling_weights, total_size = self.sampling_weights
            task_names = list(self.dataloaders.keys())
            iterators = dict((k, itertools.cycle(v)) for k, v in self.dataloaders.items())
            for i in range(total_size):
                task_name = np.random.choice(task_names, p=sampling_weights)
                yield task_name, next(iterators[task_name])
        else:
            for task_name, dataloader in self.dataloaders.items():
                for batch in dataloader:
                    yield task_name, batch

    @property
    def sampling_weights(self):
        sampling_weights = self.sizes
        total_size = sum(sampling_weights)
        Z = sum(pow(v, self.tau) for v in sampling_weights)
        sampling_weights = [pow(v, self.tau) / Z for v in sampling_weights]
        return sampling_weights, total_size

    @property
    def sizes(self):
        return [len(v) for v in self.dataloaders.values()]


class MultiTaskLearning(TorchComponent):

    def __init__(self, **kwargs) -> None:
        """ A multi-task learning (MTL) framework. It shares the same encoder across multiple decoders. These decoders
        can have dependencies on each other which will be properly handled during decoding. To integrate a component
        into this MTL framework, a component needs to implement the :class:`~hanlp.components.mtl.tasks.Task` interface.

        This framework mostly follows the architecture of :cite:`clark-etal-2019-bam`, with additional scalar mix
        tricks (:cite:`kondratyuk-straka-2019-75`) allowing each task to attend to any subset of layers. We also
        experimented with knowledge distillation on single tasks, the performance gain was nonsignificant on a large
        dataset. In the near future, we have no plan to invest more efforts in distillation, since most datasets HanLP
        uses are relatively large, and our hardware is relatively powerful.

        Args:
            **kwargs: Arguments passed to config.
        """
        super().__init__(**kwargs)
        self.model: Optional[MultiTaskModel] = None
        self.tasks: Dict[str, Task] = None
        self.vocabs = None

    def build_dataloader(self,
                         data,
                         batch_size,
                         shuffle=False,
                         device=None,
                         logger: logging.Logger = None,
                         gradient_accumulation=1,
                         tau: float = 0.8,
                         prune=None,
                         prefetch=None,
                         tasks_need_custom_eval=None,
                         cache=False,
                         debug=False,
                         **kwargs) -> DataLoader:
        # This method is only called during training or evaluation but not prediction
        dataloader = MultiTaskDataLoader(training=shuffle, tau=tau)
        for i, (task_name, task) in enumerate(self.tasks.items()):
            encoder_transform, transform = self.build_transform(task)
            training = None
            if data == 'trn':
                if debug:
                    _data = task.dev
                else:
                    _data = task.trn
                training = True
            elif data == 'dev':
                _data = task.dev
                training = False
            elif data == 'tst':
                _data = task.tst
                training = False
            else:
                _data = data
            if isinstance(data, str):
                logger.info(f'[yellow]{i + 1} / {len(self.tasks)}[/yellow] Building [blue]{data}[/blue] dataset for '
                            f'[cyan]{task_name}[/cyan] ...')
            # Adjust Tokenizer according to task config
            config = copy(task.config)
            config.pop('transform', None)
            task_dataloader: DataLoader = task.build_dataloader(_data, transform, training, device, logger,
                                                                tokenizer=encoder_transform.tokenizer,
                                                                gradient_accumulation=gradient_accumulation,
                                                                cache=isinstance(data, str), **config)
            # if prune:
            #     # noinspection PyTypeChecker
            #     task_dataset: TransformDataset = task_dataloader.dataset
            #     size_before = len(task_dataset)
            #     task_dataset.prune(prune)
            #     size_after = len(task_dataset)
            #     num_pruned = size_before - size_after
            #     logger.info(f'Pruned [yellow]{num_pruned} ({num_pruned / size_before:.1%})[/yellow] '
            #                 f'samples out of {size_before}.')
            if cache and data in ('trn', 'dev'):
                task_dataloader: CachedDataLoader = CachedDataLoader(
                    task_dataloader,
                    f'{cache}/{os.getpid()}-{data}-{task_name.replace("/", "-")}-cache.pt' if isinstance(cache,
                                                                                                         str) else None
                )
            dataloader.dataloaders[task_name] = task_dataloader
        if data == 'trn':
            sampling_weights, total_size = dataloader.sampling_weights
            headings = ['task', '#batches', '%batches', '#scaled', '%scaled', '#epoch']
            matrix = []
            min_epochs = []
            for (task_name, dataset), weight in zip(dataloader.dataloaders.items(), sampling_weights):
                epochs = len(dataset) / weight / total_size
                matrix.append(
                    [f'{task_name}', len(dataset), f'{len(dataset) / total_size:.2%}', int(total_size * weight),
                     f'{weight:.2%}', f'{epochs:.2f}'])
                min_epochs.append(epochs)
            longest = int(torch.argmax(torch.tensor(min_epochs)))
            table = markdown_table(headings, matrix)
            rows = table.splitlines()
            cells = rows[longest + 2].split('|')
            cells[-2] = cells[-2].replace(f'{min_epochs[longest]:.2f}',
                                          f'[bold][red]{min_epochs[longest]:.2f}[/red][/bold]')
            rows[longest + 2] = '|'.join(cells)
            logger.info(f'[bold][yellow]{"Samples Distribution": ^{len(rows[0])}}[/yellow][/bold]')
            logger.info('\n'.join(rows))
        if prefetch and (data == 'trn' or not tasks_need_custom_eval):
            dataloader = PrefetchDataLoader(dataloader, prefetch=prefetch)

        return dataloader

    def build_transform(self, task: Task) -> Tuple[TransformerSequenceTokenizer, TransformList]:
        encoder: ContextualWordEmbedding = self.config.encoder
        encoder_transform: TransformerSequenceTokenizer = task.build_tokenizer(encoder.transform())
        length_transform = FieldLength('token', 'token_length')
        transform = TransformList(encoder_transform, length_transform)
        extra_transform = self.config.get('transform', None)
        if extra_transform:
            transform.insert(0, extra_transform)
        return encoder_transform, transform

    def build_optimizer(self,
                        trn,
                        epochs,
                        adam_epsilon,
                        weight_decay,
                        warmup_steps,
                        lr,
                        encoder_lr,
                        **kwargs):
        model = self.model_
        encoder = model.encoder
        num_training_steps = len(trn) * epochs // self.config.get('gradient_accumulation', 1)
        encoder_parameters = list(encoder.parameters())
        parameter_groups: List[Dict[str, Any]] = []

        decoders = model.decoders
        decoder_optimizers = dict()
        for k, task in self.tasks.items():
            decoder: torch.nn.Module = decoders[k]
            decoder_parameters = list(decoder.parameters())
            if task.separate_optimizer:
                decoder_optimizers[k] = task.build_optimizer(decoder=decoder, **kwargs)
            else:
                task_lr = task.lr or lr
                parameter_groups.append({"params": decoder_parameters, 'lr': task_lr})
        parameter_groups.append({"params": encoder_parameters, 'lr': encoder_lr})
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        no_decay_parameters = set()
        for n, p in model.named_parameters():
            if any(nd in n for nd in no_decay):
                no_decay_parameters.add(p)
        no_decay_by_lr = defaultdict(list)
        for group in parameter_groups:
            _lr = group['lr']
            ps = group['params']
            group['params'] = decay_parameters = []
            group['weight_decay'] = weight_decay
            for p in ps:
                if p in no_decay_parameters:
                    no_decay_by_lr[_lr].append(p)
                else:
                    decay_parameters.append(p)
        for _lr, ps in no_decay_by_lr.items():
            parameter_groups.append({"params": ps, 'lr': _lr, 'weight_decay': 0.0})
        # noinspection PyTypeChecker
        encoder_optimizer = optimization.AdamW(
            parameter_groups,
            lr=lr,
            weight_decay=weight_decay,
            eps=adam_epsilon,
        )
        encoder_scheduler = optimization.get_linear_schedule_with_warmup(encoder_optimizer,
                                                                         num_training_steps * warmup_steps,
                                                                         num_training_steps)
        return encoder_optimizer, encoder_scheduler, decoder_optimizers

    def build_criterion(self, **kwargs):
        return dict((k, v.build_criterion(decoder=self.model_.decoders[k], **kwargs)) for k, v in self.tasks.items())

    def build_metric(self, **kwargs):
        metrics = MetricDict()
        for key, task in self.tasks.items():
            metric = task.build_metric(**kwargs)
            assert metric, f'Please implement `build_metric` of {type(task)} to return a metric.'
            metrics[key] = metric
        return metrics

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, patience=0.5, **kwargs):
        if isinstance(patience, float):
            patience = int(patience * epochs)
        best_epoch, best_metric = 0, -1
        timer = CountdownTimer(epochs)
        ratio_width = len(f'{len(trn)}/{len(trn)}')
        epoch = 0
        history = History()
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, history, ratio_width=ratio_width,
                                **self.config)
            if dev:
                self.evaluate_dataloader(dev, criterion, metric, logger, ratio_width=ratio_width, input='dev')
            report = f'{timer.elapsed_human}/{timer.total_time_human}'
            dev_score = metric.score
            if dev_score > best_metric:
                self.save_weights(save_dir)
                best_metric = dev_score
                best_epoch = epoch
                report += ' [red]saved[/red]'
            else:
                report += f' ({epoch - best_epoch})'
                if epoch - best_epoch >= patience:
                    report += ' early stop'
                    break
            timer.log(report, ratio_percentage=False, newline=True, ratio=False)
        for d in [trn, dev]:
            self._close_dataloader(d)
        if best_epoch != epoch:
            logger.info(f'Restoring best model saved [red]{epoch - best_epoch}[/red] epochs ago')
            self.load_weights(save_dir)
        return best_metric

    def _close_dataloader(self, d):
        if isinstance(d, PrefetchDataLoader):
            d.close()
            if hasattr(d.dataset, 'close'):
                self._close_dataloader(d.dataset)
        elif isinstance(d, CachedDataLoader):
            d.close()
        elif isinstance(d, MultiTaskDataLoader):
            for d in d.dataloaders.values():
                self._close_dataloader(d)

    # noinspection PyMethodOverriding
    def fit_dataloader(self,
                       trn: DataLoader,
                       criterion,
                       optimizer,
                       metric,
                       logger: logging.Logger,
                       history: History,
                       ratio_width=None,
                       gradient_accumulation=1,
                       encoder_grad_norm=None,
                       decoder_grad_norm=None,
                       patience=0.5,
                       eval_trn=False,
                       **kwargs):
        self.model.train()
        encoder_optimizer, encoder_scheduler, decoder_optimizers = optimizer
        timer = CountdownTimer(len(trn))
        total_loss = 0
        self.reset_metrics(metric)
        model = self.model_
        encoder_parameters = model.encoder.parameters()
        decoder_parameters = model.decoders.parameters()
        for idx, (task_name, batch) in enumerate(trn):
            decoder_optimizer = decoder_optimizers.get(task_name, None)
            output_dict, _ = self.feed_batch(batch, task_name)
            loss = self.compute_loss(batch, output_dict[task_name]['output'], criterion[task_name],
                                     self.tasks[task_name])
            if gradient_accumulation and gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += float(loss.item())
            if history.step(gradient_accumulation):
                if self.config.get('grad_norm', None):
                    clip_grad_norm(model, self.config.grad_norm)
                if encoder_grad_norm:
                    torch.nn.utils.clip_grad_norm_(encoder_parameters, encoder_grad_norm)
                if decoder_grad_norm:
                    torch.nn.utils.clip_grad_norm_(decoder_parameters, decoder_grad_norm)
                encoder_optimizer.step()
                encoder_optimizer.zero_grad()
                encoder_scheduler.step()
                if decoder_optimizer:
                    if isinstance(decoder_optimizer, tuple):
                        decoder_optimizer, decoder_scheduler = decoder_optimizer
                    else:
                        decoder_scheduler = None
                    decoder_optimizer.step()
                    decoder_optimizer.zero_grad()
                    if decoder_scheduler:
                        decoder_scheduler.step()
            if eval_trn:
                self.decode_output(output_dict, batch, task_name)
                self.update_metrics(batch, output_dict, metric, task_name)
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric if eval_trn else None),
                      ratio_percentage=None,
                      ratio_width=ratio_width,
                      logger=logger)
            del loss
            del output_dict
        return total_loss / timer.total

    def report_metrics(self, loss, metrics: MetricDict):
        return f'loss: {loss:.4f} {metrics.cstr()}' if metrics else f'loss: {loss:.4f}'

    # noinspection PyMethodOverriding
    @torch.no_grad()
    def evaluate_dataloader(self,
                            data: MultiTaskDataLoader,
                            criterion,
                            metric: MetricDict,
                            logger,
                            ratio_width=None,
                            input: str = None,
                            **kwargs):
        self.model.eval()
        self.reset_metrics(metric)
        tasks_need_custom_eval = self.config.get('tasks_need_custom_eval', None)
        tasks_need_custom_eval = tasks_need_custom_eval or {}
        tasks_need_custom_eval = dict((k, None) for k in tasks_need_custom_eval)
        for each in tasks_need_custom_eval:
            tasks_need_custom_eval[each] = data.dataloaders.pop(each)
        timer = CountdownTimer(len(data) + len(tasks_need_custom_eval))
        total_loss = 0
        for idx, (task_name, batch) in enumerate(data):
            output_dict, _ = self.feed_batch(batch, task_name)
            loss = self.compute_loss(batch, output_dict[task_name]['output'], criterion[task_name],
                                     self.tasks[task_name])
            total_loss += loss.item()
            self.decode_output(output_dict, batch, task_name)
            self.update_metrics(batch, output_dict, metric, task_name)
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                      logger=logger,
                      ratio_width=ratio_width)
            del loss
            del output_dict

        for task_name, dataset in tasks_need_custom_eval.items():
            task = self.tasks[task_name]
            decoder = self.model_.decoders[task_name]
            task.evaluate_dataloader(
                dataset, task.build_criterion(decoder=decoder),
                metric=metric[task_name],
                input=task.dev if input == 'dev' else task.tst,
                split=input,
                decoder=decoder,
                h=functools.partial(self._encode, task_name=task_name,
                                    cls_is_bos=task.cls_is_bos, sep_is_eos=task.sep_is_eos)
            )
            data.dataloaders[task_name] = dataset
            timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                      logger=logger,
                      ratio_width=ratio_width)

        return total_loss / timer.total, metric, data

    def build_model(self, training=False, **kwargs) -> torch.nn.Module:
        tasks = self.tasks
        encoder: ContextualWordEmbedding = self.config.encoder
        encoder_size = encoder.get_output_dim()
        scalar_mixes = torch.nn.ModuleDict()
        decoders = torch.nn.ModuleDict()
        use_raw_hidden_states = dict()
        for task_name, task in tasks.items():
            decoder = task.build_model(encoder_size, training=training, **task.config)
            assert decoder, f'Please implement `build_model` of {type(task)} to return a decoder.'
            decoders[task_name] = decoder
            if task.scalar_mix:
                scalar_mix = task.scalar_mix.build()
                scalar_mixes[task_name] = scalar_mix
                # Activate scalar mix starting from 0-th layer
                encoder.scalar_mix = 0
            use_raw_hidden_states[task_name] = task.use_raw_hidden_states
        encoder.ret_raw_hidden_states = any(use_raw_hidden_states.values())
        return MultiTaskModel(encoder.module(training=training), scalar_mixes, decoders, use_raw_hidden_states)

    def predict(self,
                data: Union[str, List[str]],
                batch_size: int = None,
                tasks: Optional[Union[str, List[str]]] = None,
                skip_tasks: Optional[Union[str, List[str]]] = None,
                resolved_tasks=None,
                **kwargs) -> Document:
        """Predict on data.

        Args:
            data: A sentence or a list of sentences.
            batch_size: Decoding batch size.
            tasks: The tasks to predict.
            skip_tasks: The tasks to skip.
            resolved_tasks: The resolved tasks to override ``tasks`` and ``skip_tasks``.
            **kwargs: Not used.

        Returns:
            A :class:`~hanlp_common.document.Document`.
        """
        doc = Document()
        if not data:
            return doc

        target_tasks = resolved_tasks or self.resolve_tasks(tasks, skip_tasks)
        flatten_target_tasks = [self.tasks[t] for group in target_tasks for t in group]
        cls_is_bos = any([x.cls_is_bos for x in flatten_target_tasks])
        sep_is_eos = any([x.sep_is_eos for x in flatten_target_tasks])
        # Now build the dataloaders and execute tasks
        first_task_name: str = list(target_tasks[0])[0]
        first_task: Task = self.tasks[first_task_name]
        encoder_transform, transform = self.build_transform(first_task)
        # Override the tokenizer config of the 1st task
        encoder_transform.sep_is_eos = sep_is_eos
        encoder_transform.cls_is_bos = cls_is_bos
        average_subwords = self.model.encoder.average_subwords
        flat = first_task.input_is_flat(data)
        if flat:
            data = [data]
        device = self.device
        samples = first_task.build_samples(data, cls_is_bos=cls_is_bos, sep_is_eos=sep_is_eos)
        dataloader = first_task.build_dataloader(samples, transform=transform, device=device)
        results = defaultdict(list)
        order = []
        for batch in dataloader:
            order.extend(batch[IDX])
            # Run the first task, let it make the initial batch for the successors
            output_dict = self.predict_task(first_task, first_task_name, batch, results, run_transform=True,
                                            cls_is_bos=cls_is_bos, sep_is_eos=sep_is_eos)
            # Run each task group in order
            for group_id, group in enumerate(target_tasks):
                # We could parallelize this in the future
                for task_name in group:
                    if task_name == first_task_name:
                        continue
                    output_dict = self.predict_task(self.tasks[task_name], task_name, batch, results, output_dict,
                                                    run_transform=True, cls_is_bos=cls_is_bos, sep_is_eos=sep_is_eos)
                if group_id == 0:
                    # We are kind of hard coding here. If the first task is a tokenizer,
                    # we need to convert the hidden and mask to token level
                    if first_task_name.startswith('tok'):
                        spans = []
                        tokens = []
                        for span_per_sent, token_per_sent in zip(output_dict[first_task_name]['prediction'],
                                                                 results[first_task_name][-len(batch[IDX]):]):
                            if cls_is_bos:
                                span_per_sent = [(-1, 0)] + span_per_sent
                                token_per_sent = [BOS] + token_per_sent
                            if sep_is_eos:
                                span_per_sent = span_per_sent + [(span_per_sent[-1][0] + 1, span_per_sent[-1][1] + 1)]
                                token_per_sent = token_per_sent + [EOS]
                            # The offsets start with 0 while [CLS] is zero
                            if average_subwords:
                                span_per_sent = [list(range(x[0] + 1, x[1] + 1)) for x in span_per_sent]
                            else:
                                span_per_sent = [x[0] + 1 for x in span_per_sent]
                            spans.append(span_per_sent)
                            tokens.append(token_per_sent)
                        spans = PadSequenceDataLoader.pad_data(spans, 0, torch.long, device=device)
                        output_dict['hidden'] = pick_tensor_for_each_token(output_dict['hidden'], spans,
                                                                           average_subwords)
                        batch['token_token_span'] = spans
                        batch['token'] = tokens
                        # noinspection PyTypeChecker
                        batch['token_length'] = torch.tensor([len(x) for x in tokens], dtype=torch.long, device=device)
                        batch.pop('mask', None)
        # Put results into doc in the order of tasks
        for k in self.config.task_names:
            v = results.get(k, None)
            if v is None:
                continue
            doc[k] = reorder(v, order)
        # Allow task to perform finalization on document
        for group in target_tasks:
            for task_name in group:
                task = self.tasks[task_name]
                task.finalize_document(doc, task_name)
        # If no tok in doc, use raw input as tok
        if not any(k.startswith('tok') for k in doc):
            doc['tok'] = data
        if flat:
            for k, v in list(doc.items()):
                doc[k] = v[0]
        # If there is only one field, don't bother to wrap it
        # if len(doc) == 1:
        #     return list(doc.values())[0]
        return doc

    def resolve_tasks(self, tasks, skip_tasks) -> List[Iterable[str]]:
        # Now we decide which tasks to perform and their orders
        tasks_in_topological_order = self._tasks_in_topological_order
        task_topological_order = self._task_topological_order
        computation_graph = self._computation_graph
        target_tasks = self._resolve_task_name(tasks)
        if not target_tasks:
            target_tasks = tasks_in_topological_order
        else:
            target_topological_order = defaultdict(set)
            for task_name in target_tasks:
                for dependency in topological_sort(computation_graph, task_name):
                    target_topological_order[task_topological_order[dependency]].add(dependency)
            target_tasks = [item[1] for item in sorted(target_topological_order.items())]
        if skip_tasks:
            skip_tasks = self._resolve_task_name(skip_tasks)
            target_tasks = [x - skip_tasks for x in target_tasks]
            target_tasks = [x for x in target_tasks if x]
        assert target_tasks, f'No task to perform due to `tasks = {tasks}`.'
        # Sort target tasks within the same group in a defined order
        target_tasks = [sorted(x, key=lambda _x: self.config.task_names.index(_x)) for x in target_tasks]
        return target_tasks

    def predict_task(self, task: Task, output_key, batch, results, output_dict=None, run_transform=True,
                     cls_is_bos=True, sep_is_eos=True):
        output_dict, batch = self.feed_batch(batch, output_key, output_dict, run_transform, cls_is_bos, sep_is_eos,
                                             results)
        self.decode_output(output_dict, batch, output_key)
        results[output_key].extend(task.prediction_to_result(output_dict[output_key]['prediction'], batch))
        return output_dict

    def _resolve_task_name(self, dependencies):
        resolved_dependencies = set()
        if isinstance(dependencies, str):
            if dependencies in self.tasks:
                resolved_dependencies.add(dependencies)
            elif dependencies.endswith('*'):
                resolved_dependencies.update(x for x in self.tasks if x.startswith(dependencies[:-1]))
            else:
                prefix_matched = prefix_match(dependencies, self.config.task_names)
                assert prefix_matched, f'No prefix matching for {dependencies}. ' \
                                       f'Check your dependencies definition: {list(self.tasks.values())}'
                resolved_dependencies.add(prefix_matched)
        elif isinstance(dependencies, Iterable):
            resolved_dependencies.update(set(chain.from_iterable(self._resolve_task_name(x) for x in dependencies)))
        return resolved_dependencies

    def fit(self,
            encoder: Embedding,
            tasks: Dict[str, Task],
            save_dir,
            epochs,
            patience=0.5,
            lr=1e-3,
            encoder_lr=5e-5,
            adam_epsilon=1e-8,
            weight_decay=0.0,
            warmup_steps=0.1,
            gradient_accumulation=1,
            grad_norm=5.0,
            encoder_grad_norm=None,
            decoder_grad_norm=None,
            tau: float = 0.8,
            transform=None,
            # prune: Callable = None,
            eval_trn=True,
            prefetch=None,
            tasks_need_custom_eval=None,
            _device_placeholder=False,
            cache=False,
            devices=None,
            logger=None,
            seed=None,
            **kwargs):
        trn_data, dev_data, batch_size = 'trn', 'dev', None
        task_names = list(tasks.keys())
        return super().fit(**merge_locals_kwargs(locals(), kwargs, excludes=('self', 'kwargs', '__class__', 'tasks')),
                           **tasks)

    # noinspection PyAttributeOutsideInit
    def on_config_ready(self, **kwargs):
        self.tasks = dict((key, task) for key, task in self.config.items() if isinstance(task, Task))
        computation_graph = dict()
        for task_name, task in self.tasks.items():
            dependencies = task.dependencies
            resolved_dependencies = self._resolve_task_name(dependencies)
            computation_graph[task_name] = resolved_dependencies

        # We can cache this order
        tasks_in_topological_order = list(toposort(computation_graph))
        task_topological_order = dict()
        for i, group in enumerate(tasks_in_topological_order):
            for task_name in group:
                task_topological_order[task_name] = i
        self._tasks_in_topological_order = tasks_in_topological_order
        self._task_topological_order = task_topological_order
        self._computation_graph = computation_graph

    @staticmethod
    def reset_metrics(metrics: Dict[str, Metric]):
        for metric in metrics.values():
            metric.reset()

    def feed_batch(self,
                   batch: Dict[str, Any],
                   task_name,
                   output_dict=None,
                   run_transform=False,
                   cls_is_bos=False,
                   sep_is_eos=False,
                   results=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        h, output_dict = self._encode(batch, task_name, output_dict, cls_is_bos, sep_is_eos)
        task = self.tasks[task_name]
        if run_transform:
            batch = task.transform_batch(batch, results=results, cls_is_bos=cls_is_bos, sep_is_eos=sep_is_eos)
        batch['mask'] = mask = util.lengths_to_mask(batch['token_length'])
        output_dict[task_name] = {
            'output': task.feed_batch(h,
                                      batch=batch,
                                      mask=mask,
                                      decoder=self.model.decoders[task_name]),
            'mask': mask
        }
        return output_dict, batch

    def _encode(self, batch, task_name, output_dict=None, cls_is_bos=False, sep_is_eos=False):
        model = self.model
        if output_dict:
            hidden, raw_hidden = output_dict['hidden'], output_dict['raw_hidden']
        else:
            hidden = model.encoder(batch)
            if isinstance(hidden, tuple):
                hidden, raw_hidden = hidden
            else:
                raw_hidden = None
            output_dict = {'hidden': hidden, 'raw_hidden': raw_hidden}
        hidden_states = raw_hidden if model.use_raw_hidden_states[task_name] else hidden
        if task_name in model.scalar_mixes:
            scalar_mix = model.scalar_mixes[task_name]
            h = scalar_mix(hidden_states)
        else:
            if model.scalar_mixes:  # If any task enables scalar_mix, hidden_states will be a 4d tensor
                hidden_states = hidden_states[-1, :, :, :]
            h = hidden_states
        # If the task doesn't need cls while h has cls, remove cls
        task = self.tasks[task_name]
        if cls_is_bos and not task.cls_is_bos:
            h = h[:, 1:, :]
        if sep_is_eos and not task.sep_is_eos:
            h = h[:, :-1, :]
        return h, output_dict

    def decode_output(self, output_dict, batch, task_name=None):
        if not task_name:
            for task_name, task in self.tasks.items():
                output_per_task = output_dict.get(task_name, None)
                if output_per_task is not None:
                    output_per_task['prediction'] = task.decode_output(
                        output_per_task['output'],
                        output_per_task['mask'],
                        batch, self.model.decoders[task_name])
        else:
            output_per_task = output_dict[task_name]
            output_per_task['prediction'] = self.tasks[task_name].decode_output(
                output_per_task['output'],
                output_per_task['mask'],
                batch,
                self.model.decoders[task_name])

    def update_metrics(self, batch: Dict[str, Any], output_dict: Dict[str, Any], metrics: MetricDict, task_name):
        task = self.tasks[task_name]
        output_per_task = output_dict.get(task_name, None)
        if output_per_task:
            output = output_per_task['output']
            prediction = output_per_task['prediction']
            metric = metrics.get(task_name, None)
            task.update_metrics(batch, output, prediction, metric)

    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion: Callable,
                     task: Task) -> torch.FloatTensor:
        return task.compute_loss(batch, output, criterion)

    def evaluate(self, save_dir=None, logger: logging.Logger = None, batch_size=None, output=False, **kwargs):
        rets = super().evaluate('tst', save_dir, logger, batch_size, output, **kwargs)
        tst = rets[-1]
        self._close_dataloader(tst)
        return rets

    def save_vocabs(self, save_dir, filename='vocabs.json'):
        for task_name, task in self.tasks.items():
            task.save_vocabs(save_dir, f'{task_name}_{filename}')

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        for task_name, task in self.tasks.items():
            task.load_vocabs(save_dir, f'{task_name}_{filename}')

    def parallelize(self, devices: List[Union[int, torch.device]]):
        raise NotImplementedError('Parallelization is not implemented yet.')

    def __call__(self, data, batch_size=None, **kwargs) -> Document:
        return super().__call__(data, batch_size, **kwargs)

    def __getitem__(self, task_name: str) -> Task:
        return self.tasks[task_name]

    def __delitem__(self, task_name: str):
        """Delete a task (and every resource it owns) from this component.

        Args:
            task_name: The name of the task to be deleted.

        Examples:
            >>> del mtl['dep']  # Delete dep from MTL

        """
        del self.config[task_name]
        self.config.task_names.remove(task_name)
        del self.tasks[task_name]
        del self.model.decoders[task_name]
        del self._computation_graph[task_name]
        self._task_topological_order.pop(task_name)
        for group in self._tasks_in_topological_order:
            group: set = group
            group.discard(task_name)

    def __repr__(self):
        return repr(self.config)

    def items(self):
        yield from self.tasks.items()
