# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-22 23:40
import logging
from typing import Union, List, Callable, Any, Dict

import torch
from torch.utils.data import DataLoader

from hanlp.common.dataset import PadSequenceDataLoader
from hanlp.common.structure import History
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import FieldLength
from hanlp.common.vocab import Vocab
from hanlp.components.ner.biaffine_ner.biaffine_ner import BiaffineNamedEntityRecognizer
from hanlp.components.parsers.hpsg import trees, bracket_eval, dep_eval
from hanlp.components.parsers.hpsg.bracket_eval import FScore
from hanlp.components.parsers.hpsg.dep_eval import SimpleAttachmentScore
from hanlp.components.parsers.hpsg.hpsg_dataset import HeadDrivenPhraseStructureDataset
from hanlp.components.parsers.hpsg.hpsg_parser_model import ChartParser
from hanlp.datasets.parsing.conll_dataset import append_bos_eos
from hanlp.layers.embeddings.embedding import Embedding
from hanlp.metrics.parsing.attachmentscore import AttachmentScore
from hanlp.utils.time_util import CountdownTimer
from hanlp_common.util import merge_locals_kwargs


class HeadDrivenPhraseStructureParser(TorchComponent):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: ChartParser = None

    # noinspection PyCallByClass
    def build_optimizer(self,
                        trn,
                        epochs,
                        lr,
                        adam_epsilon,
                        weight_decay,
                        warmup_steps,
                        transformer_lr,
                        **kwargs):
        return BiaffineNamedEntityRecognizer.build_optimizer(self,
                                                             trn,
                                                             epochs,
                                                             lr,
                                                             adam_epsilon,
                                                             weight_decay,
                                                             warmup_steps,
                                                             transformer_lr)

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs):
        return AttachmentScore()

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        model = ChartParser(self.config.embed.module(vocabs=self.vocabs),
                            self.vocabs.pos, self.vocabs.label, self.vocabs.rel, self.config)
        return model

    # noinspection PyMethodOverriding
    def build_dataloader(self, data, batch_size, shuffle, device, logger: logging.Logger, sampler_builder,
                         gradient_accumulation,
                         **kwargs) -> DataLoader:
        # shuffle = False  # We need to find the smallest grad_acc
        dataset = HeadDrivenPhraseStructureDataset(data, transform=[append_bos_eos])
        if self.config.get('transform', None):
            dataset.append_transform(self.config.transform)
        dataset.append_transform(self.vocabs)
        if isinstance(self.config.embed, Embedding):
            transform = self.config.embed.transform(vocabs=self.vocabs)
            if transform:
                dataset.append_transform(transform)
        dataset.append_transform(self.vocabs)
        field_length = FieldLength('token')
        dataset.append_transform(field_length)
        if isinstance(data, str):
            dataset.purge_cache()  # Enable cache
        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger)
        if 'token' in self.vocabs:
            lens = [x[field_length.dst] for x in dataset]
        else:
            lens = [len(x['token_input_ids']) for x in dataset]
        if sampler_builder:
            sampler = sampler_builder.build(lens, shuffle, gradient_accumulation)
        else:
            sampler = None
        return PadSequenceDataLoader(batch_sampler=sampler,
                                     batch_size=batch_size,
                                     device=device,
                                     dataset=dataset)

    def predict(self, data: Union[str, List[str]], batch_size: int = None, **kwargs):
        pass

    def build_vocabs(self, dataset, logger, **kwargs):
        self.vocabs.rel = Vocab(pad_token=None, unk_token=None)
        self.vocabs.pos = Vocab(pad_token=None, unk_token=None)
        self.vocabs.label = label_vocab = Vocab(pad_token='', unk_token=None)
        label_vocab.add(trees.Sub_Head)
        for each in dataset:
            tree = each['hpsg']
            nodes = [tree]
            while nodes:
                node = nodes.pop()
                if isinstance(node, trees.InternalParseNode):
                    label_vocab.add('\t'.join(node.label))
                    nodes.extend(reversed(node.children))
        self.vocabs['rel'].set_unk_as_safe_unk()
        label_vocab.set_unk_as_safe_unk()
        self.vocabs.lock()
        self.vocabs.summary(logger)

    def fit(self, trn_data, dev_data, save_dir,
            embed: Embedding,
            batch_size=100,
            epochs=100,
            sampler='sorting',
            n_buckets=32,
            batch_max_tokens=None,
            sampler_builder=None,
            attention_dropout=0.2,
            bert_do_lower_case=True,
            bert_model='bert-large-uncased',
            bert_transliterate='',
            char_lstm_input_dropout=0.2,
            clip_grad_norm=0.0,
            const_lada=0.5,
            d_biaffine=1024,
            d_char_emb=64,
            d_ff=2048,
            d_kv=64,
            d_label_hidden=250,
            d_model=1024,
            dataset='ptb',
            elmo_dropout=0.5,
            embedding_dropout=0.2,
            embedding_path='data/glove.gz',
            embedding_type='random',
            lal_combine_as_self=False,
            lal_d_kv=128,
            lal_d_proj=128,
            lal_partitioned=True,
            lal_pwff=True,
            lal_q_as_matrix=False,
            lal_resdrop=False,
            max_len_dev=0,
            max_len_train=0,
            morpho_emb_dropout=0.2,
            num_heads=8,
            num_layers=3,
            pad_left=False,
            partitioned=True,
            relu_dropout=0.2,
            residual_dropout=0.2,
            sentence_max_len=300,
            step_decay=True,
            step_decay_factor=0.5,
            step_decay_patience=5,
            tag_emb_dropout=0.2,
            timing_dropout=0.0,
            dont_use_encoder=False,
            use_cat=False,
            use_chars_lstm=False,
            use_elmo=False,
            use_lal=True,
            use_tags=True,
            use_words=False,
            word_emb_dropout=0.4,
            lr=1e-3,
            transformer_lr=5e-5,
            adam_epsilon=1e-6,
            weight_decay=0.01,
            warmup_steps=0.1,
            grad_norm=5.0,
            gradient_accumulation=1,
            devices=None, logger=None, seed=None, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    # noinspection PyMethodOverriding
    def fit_dataloader(self,
                       trn: DataLoader,
                       criterion,
                       optimizer,
                       metric,
                       logger: logging.Logger,
                       history: History,
                       linear_scheduler=None,
                       gradient_accumulation=1,
                       **kwargs):
        self.model.train()
        timer = CountdownTimer(history.num_training_steps(len(trn), gradient_accumulation))
        total_loss = 0
        self.reset_metrics(metric)
        for idx, batch in enumerate(trn):
            output_dict = self.feed_batch(batch)
            self.update_metrics(batch, output_dict, metric)
            loss = output_dict['loss']
            if gradient_accumulation > 1:
                loss /= gradient_accumulation
            loss.backward()
            total_loss += loss.item()
            if history.step(gradient_accumulation):
                self._step(optimizer, linear_scheduler)
                timer.log(self.report_metrics(total_loss / (timer.current + 1), metric), ratio_percentage=None,
                          logger=logger)
            del loss
        return total_loss / timer.total

    def _step(self, optimizer, linear_scheduler):
        if self.config.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        if linear_scheduler:
            linear_scheduler.step()

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
        self.reset_metrics(metric)
        timer = CountdownTimer(len(data))
        gold_tree = []
        pred_tree = []
        pred_head = []
        pred_type = []
        gold_type = []
        gold_word = []
        gold_pos = []
        gold_head = []
        for batch in data:
            output_dict = self.feed_batch(batch)
            gold_tree += batch['tree']
            pred_tree += output_dict['predicted_tree']

            pred_head += output_dict['pred_head']
            pred_type += output_dict['pred_type']

            gold_type += batch['DEPREL']
            gold_head += batch['HEAD']
            gold_pos += batch['CPOS']
            gold_word += batch['FORM']
            assert len(pred_head) == len(gold_head)
            self.update_metrics(batch, output_dict, metric)
            timer.log('', ratio_percentage=None, ratio_width=ratio_width)

        tree_score: FScore = bracket_eval.evalb(gold_tree, pred_tree)
        assert len(pred_head) == len(pred_type)
        assert len(pred_type) == len(gold_type)
        lens = [len(x) for x in gold_word]
        stats, stats_nopunc, stats_root, test_total_inst = dep_eval.eval(len(pred_head), gold_word, gold_pos,
                                                                         pred_head,
                                                                         pred_type, gold_head, gold_type,
                                                                         lens, punct_set=None,
                                                                         symbolic_root=False)

        test_ucorrect, test_lcorrect, test_total, test_ucomlpete_match, test_lcomplete_match = stats
        test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc, test_ucomlpete_match_nopunc, test_lcomplete_match_nopunc = stats_nopunc
        test_root_correct, test_total_root = stats_root
        dep_score = SimpleAttachmentScore(test_ucorrect_nopunc / test_total_nopunc,
                                          test_lcorrect_nopunc / test_total_nopunc)
        timer.log(f'{tree_score} {dep_score}', ratio_percentage=None, ratio_width=ratio_width, logger=logger)
        return tree_score, dep_score

    def reset_metrics(self, metrics):
        pass
        # for m in metrics:
        #     m.reset()

    def report_metrics(self, loss, metrics):
        return f'loss: {loss:.4f}'

    def feed_batch(self, batch) -> Dict[str, Any]:
        predicted_tree, loss_or_score = self.model(batch)
        outputs = {}
        if isinstance(loss_or_score, torch.Tensor):
            loss_or_score /= len(batch['hpsg'])
            loss = loss_or_score
            outputs['loss'] = loss
        else:
            score = loss_or_score
            outputs['score'] = score
        if predicted_tree:
            predicted_tree = [p.convert() for p in predicted_tree]
            pred_head = [[leaf.father for leaf in tree.leaves()] for tree in predicted_tree]
            pred_type = [[leaf.type for leaf in tree.leaves()] for tree in predicted_tree]
            outputs.update({
                'predicted_tree': predicted_tree,
                'pred_head': pred_head,
                'pred_type': pred_type
            }),
        return outputs

    def update_metrics(self, batch: dict, output_dict: dict, metrics):
        pass
        # assert len(output_dict['prediction']) == len(batch['ner'])
        # for pred, gold in zip(output_dict['prediction'], batch['ner']):
        #     metrics(set(pred), set(gold))

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
        best_epoch, best_score = 0, -1
        optimizer, scheduler = optimizer
        timer = CountdownTimer(epochs)
        _len_trn = len(trn) // self.config.gradient_accumulation
        ratio_width = len(f'{_len_trn}/{_len_trn}')
        history = History()
        for epoch in range(1, epochs + 1):
            logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
            self.fit_dataloader(trn, criterion, optimizer, metric, logger, history,
                                linear_scheduler=scheduler if self.use_transformer else None, **kwargs)
            if dev:
                metric = self.evaluate_dataloader(dev, criterion, metric, logger, ratio_width=ratio_width)
            report = f'{timer.elapsed_human}/{timer.total_time_human}'
            dev_score = sum(x.score for x in metric) / len(metric)
            if not self.use_transformer:
                scheduler.step(dev_score)
            if dev_score > best_score:
                self.save_weights(save_dir)
                best_score = dev_score
                report += ' [red]saved[/red]'
            timer.log(report, ratio_percentage=False, newline=True, ratio=False)

    @property
    def use_transformer(self):
        return 'token' not in self.vocabs

    def _get_transformer(self):
        return getattr(self.model.embed, 'transformer', None)
