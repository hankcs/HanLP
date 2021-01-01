# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-28 15:12
import functools
from collections import Counter
from typing import Union, List

import torch
from torch import nn

from hanlp_common.constant import UNK
from hanlp.common.transform import TransformList
from hanlp.components.parsers.biaffine.biaffine_dep import BiaffineDependencyParser
from hanlp_common.conll import CoNLLUWord, CoNLLSentence
from hanlp.datasets.parsing.semeval15 import unpack_deps_to_head_deprel, append_bos_to_form_pos
from hanlp.metrics.parsing.labeled_f1 import LabeledF1
from hanlp_common.util import merge_locals_kwargs


class BiaffineSemanticDependencyParser(BiaffineDependencyParser):
    def __init__(self) -> None:
        """Implementation of "Stanford's graph-based neural dependency parser at
        the conll 2017 shared task" (:cite:`dozat2017stanford`).
        """
        super().__init__()

    def get_pad_dict(self):
        return {'arc': False}

    def build_metric(self, **kwargs):
        return LabeledF1()

    # noinspection PyMethodOverriding
    def build_dataset(self, data, transform=None):
        transforms = TransformList(functools.partial(append_bos_to_form_pos, pos_key='UPOS'),
                                   functools.partial(unpack_deps_to_head_deprel, pad_rel=self.config.pad_rel))
        if transform:
            transforms.append(transform)
        return super(BiaffineSemanticDependencyParser, self).build_dataset(data, transforms)

    def build_criterion(self, **kwargs):
        return nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()

    def feed_batch(self, batch):
        arc_scores, rel_scores, mask, puncts = super().feed_batch(batch)
        mask = self.convert_to_3d_mask(arc_scores, mask)
        puncts = self.convert_to_3d_puncts(puncts, mask)
        return arc_scores, rel_scores, mask, puncts

    @staticmethod
    def convert_to_3d_puncts(puncts, mask):
        if puncts is not None:
            puncts = puncts.unsqueeze(-1).expand_as(mask)
        return puncts

    @staticmethod
    def convert_to_3d_mask(arc_scores, mask):
        # 3d masks
        mask = mask.unsqueeze(-1).expand_as(arc_scores)
        mask = mask & mask.transpose(1, 2)
        return mask

    def compute_loss(self, arc_scores, rel_scores, arcs, rels, mask: torch.BoolTensor, criterion, batch=None):
        bce, ce = criterion
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores, rels = rel_scores[arcs], rels[arcs]
        arc_loss = bce(arc_scores, arcs.to(torch.float))
        arc_loss_interpolation = self.config.get('arc_loss_interpolation', None)
        loss = arc_loss * arc_loss_interpolation if arc_loss_interpolation else arc_loss
        if len(rels):
            rel_loss = ce(rel_scores, rels)
            loss += (rel_loss * (1 - arc_loss_interpolation)) if arc_loss_interpolation else rel_loss
        if arc_loss_interpolation:
            loss *= 2
        return loss

    def cache_dataset(self, dataset, timer, training=False, logger=None):
        if not self.config.apply_constraint:
            return super(BiaffineSemanticDependencyParser, self).cache_dataset(dataset, timer, training)
        num_roots = Counter()
        no_zero_head = True
        root_rels = Counter()
        for each in dataset:
            if training:
                num_roots[sum([x[0] for x in each['arc']])] += 1
                no_zero_head &= all([x != '_' for x in each['DEPS']])
                head_is_root = [i for i in range(len(each['arc'])) if each['arc'][i][0]]
                if head_is_root:
                    for i in head_is_root:
                        root_rels[each['rel'][i][0]] += 1
            timer.log('Preprocessing and caching samples [blink][yellow]...[/yellow][/blink]')
        if training:
            if self.config.single_root is None:
                self.config.single_root = len(num_roots) == 1 and num_roots.most_common()[0][0] == 1
            if self.config.no_zero_head is None:
                self.config.no_zero_head = no_zero_head
            root_rel = root_rels.most_common()[0][0]
            self.config.root_rel_id = self.vocabs['rel'].get_idx(root_rel)
            if logger:
                logger.info(f'Training set properties: [blue]single_root = {self.config.single_root}[/blue], '
                            f'[blue]no_zero_head = {no_zero_head}[/blue], '
                            f'[blue]root_rel = {root_rel}[/blue]')

    def decode(self, arc_scores, rel_scores, mask, batch=None):
        eye = torch.arange(0, arc_scores.size(1), device=arc_scores.device).view(1, 1, -1).expand(
            arc_scores.size(0), -1, -1)
        inf = float('inf')
        arc_scores.scatter_(dim=1, index=eye, value=-inf)

        if self.config.apply_constraint:
            if self.config.get('single_root', False):
                root_mask = arc_scores[:, :, 0].argmax(dim=-1).unsqueeze_(-1).expand_as(arc_scores[:, :, 0])
                arc_scores[:, :, 0] = -inf
                arc_scores[:, :, 0].scatter_(dim=-1, index=root_mask, value=inf)

            root_rel_id = self.config.root_rel_id
            rel_scores[:, :, 0, root_rel_id] = inf
            rel_scores[:, :, 1:, root_rel_id] = -inf

            arc_scores_T = arc_scores.transpose(-1, -2)
            arc = ((arc_scores > 0) & (arc_scores_T < arc_scores))
            if self.config.get('no_zero_head', False):
                arc_scores_fix = arc_scores_T.argmax(dim=-2).unsqueeze_(-1).expand_as(arc_scores)
                arc.scatter_(dim=-1, index=arc_scores_fix, value=True)
        else:
            arc = arc_scores > 0
        rel = rel_scores.argmax(dim=-1)
        return arc, rel

    def collect_outputs_extend(self, predictions, arc_preds, rel_preds, lens, mask):
        predictions.extend(zip(arc_preds.tolist(), rel_preds.tolist(), mask.tolist()))
        # all_arcs.extend(seq.tolist() for seq in arc_preds[mask].split([x * x for x in lens]))
        # all_rels.extend(seq.tolist() for seq in rel_preds[mask].split([x * x for x in lens]))

    def predictions_to_human(self, predictions, outputs, data, use_pos):
        for d, (arcs, rels, masks) in zip(data, predictions):
            sent = CoNLLSentence()
            for idx, (cell, a, r) in enumerate(zip(d, arcs[1:], rels[1:])):
                if use_pos:
                    token, pos = cell
                else:
                    token, pos = cell, None
                heads = [i for i in range(len(d) + 1) if a[i]]
                deprels = [self.vocabs['rel'][r[i]] for i in range(len(d) + 1) if a[i]]
                sent.append(
                    CoNLLUWord(idx + 1, token, upos=pos, head=None, deprel=None, deps=list(zip(heads, deprels))))
            outputs.append(sent)

    def fit(self, trn_data, dev_data, save_dir,
            feat=None,
            n_embed=100,
            pretrained_embed=None,
            transformer=None,
            average_subwords=False,
            word_dropout: float = 0.2,
            transformer_hidden_dropout=None,
            layer_dropout=0,
            mix_embedding: int = None,
            embed_dropout=.33,
            n_lstm_hidden=400,
            n_lstm_layers=3,
            hidden_dropout=.33,
            n_mlp_arc=500,
            n_mlp_rel=100,
            mlp_dropout=.33,
            arc_dropout=None,
            rel_dropout=None,
            arc_loss_interpolation=0.4,
            lr=2e-3,
            transformer_lr=5e-5,
            mu=.9,
            nu=.9,
            epsilon=1e-12,
            clip=5.0,
            decay=.75,
            decay_steps=5000,
            weight_decay=0,
            warmup_steps=0.1,
            separate_optimizer=True,
            patience=100,
            batch_size=None,
            sampler_builder=None,
            lowercase=False,
            epochs=50000,
            apply_constraint=False,
            single_root=None,
            no_zero_head=None,
            punct=False,
            min_freq=2,
            logger=None,
            verbose=True,
            unk=UNK,
            pad_rel=None,
            max_sequence_length=512,
            gradient_accumulation=1,
            devices: Union[float, int, List[int]] = None,
            transform=None,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))
