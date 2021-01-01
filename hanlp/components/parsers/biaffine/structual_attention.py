# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-26 10:40
from typing import Union, List

import torch
import torch.nn.functional as F
from alnlp.modules.util import lengths_to_mask
from torch import nn

from hanlp.common.torch_component import TorchComponent
from hanlp.components.parsers.biaffine.biaffine_dep import BiaffineDependencyParser
from hanlp.components.parsers.biaffine.biaffine_model import BiaffineDecoder
from hanlp.layers.transformers.encoder import TransformerEncoder
from hanlp.layers.transformers.pt_imports import PreTrainedModel, PreTrainedTokenizer
from hanlp.metrics.accuracy import CategoricalAccuracy
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp_common.util import merge_locals_kwargs


class StructuralAttentionLayer(nn.Module):

    def __init__(self, hidden_size, n_mlp_arc, n_mlp_rel, mlp_dropout, n_rels, projeciton=None) -> None:
        super().__init__()
        self.biaffine = BiaffineDecoder(hidden_size, n_mlp_arc, n_mlp_rel, mlp_dropout, n_rels)
        if projeciton:
            self.projection = nn.Linear(hidden_size, projeciton)
            hidden_size = projeciton
        else:
            self.projection = None
        self.head_WV = nn.Parameter(torch.randn(n_rels, hidden_size, hidden_size))
        self.dense = nn.Linear(hidden_size * n_rels, hidden_size)
        self.activation = nn.GELU()

    def forward(self, x, mask):
        s_arc, s_rel = self.biaffine(x, mask)
        p_arc = F.softmax(s_arc, dim=-1) * mask.unsqueeze(-1)
        p_rel = F.softmax(s_rel, -1)
        A = p_arc.unsqueeze(-1) * p_rel
        if self.projection:
            x = self.projection(x)
        Ax = torch.einsum('bijk,bih->bihk', A, x)
        AxW = torch.einsum('bihk,khm->bihk', Ax, self.head_WV)
        AxW = AxW.flatten(2)
        x = self.dense(AxW)
        x = self.activation(x)
        return s_arc, s_rel, x


class StructuralAttentionModel(nn.Module):
    def __init__(self,
                 config,
                 transformer: PreTrainedModel = None,
                 transformer_tokenizer: PreTrainedTokenizer = None
                 ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(transformer,
                                          transformer_tokenizer,
                                          config.average_subwords,
                                          config.scalar_mix,
                                          None,  # No word_dropout since SA is predicting masked tokens
                                          config.transformer_hidden_dropout,
                                          config.layer_dropout,
                                          config.max_sequence_length)
        hidden_size = transformer.config.hidden_size
        self.sa = StructuralAttentionLayer(hidden_size,
                                           config.n_mlp_arc,
                                           config.n_mlp_rel,
                                           config.mlp_dropout,
                                           config.n_rels,
                                           config.projection
                                           )
        if config.projection:
            hidden_size = config.projection
        self.mlm = nn.Linear(hidden_size, transformer_tokenizer.vocab_size)

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask=None,
                token_type_ids=None,
                token_span=None,
                mask=None,
                batch=None,
                **kwargs):
        h = self.encoder(input_ids, attention_mask, token_type_ids, token_span)
        s_arc, s_rel, h = self.sa(h, mask)
        x = self.mlm(h)
        return s_arc, s_rel, x


class MaskedTokenGenerator(object):

    def __init__(self, transformer_tokenizer: PreTrainedTokenizer, mask_prob=0.15) -> None:
        super().__init__()
        self.mask_prob = mask_prob
        self.transformer_tokenizer = transformer_tokenizer
        self.oov = transformer_tokenizer.mask_token_id
        self.pad = transformer_tokenizer.pad_token_id
        self.cls = transformer_tokenizer.cls_token_id
        self.sep = transformer_tokenizer.sep_token_id
        self.excludes = [self.pad, self.cls, self.sep]

    def __call__(self, tokens: torch.LongTensor, prefix_mask: torch.LongTensor):
        padding_mask = tokens.new_ones(tokens.size(), dtype=torch.bool)
        for pad in self.excludes:
            padding_mask &= (tokens != pad)
        padding_mask &= prefix_mask  # Only mask prefixes since the others won't be attended
        # Create a uniformly random mask selecting either the original words or OOV tokens
        dropout_mask = (tokens.new_empty(tokens.size(), dtype=torch.float).uniform_() < self.mask_prob)
        oov_mask = dropout_mask & padding_mask

        oov_fill = tokens.new_empty(tokens.size(), dtype=torch.long).fill_(self.oov)

        result = torch.where(oov_mask, oov_fill, tokens)
        return result, oov_mask


class StructuralAttentionParser(BiaffineDependencyParser):
    def __init__(self) -> None:
        super().__init__()
        self.model: StructuralAttentionModel = None
        self.mlm_generator: MaskedTokenGenerator = None

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        transformer = TransformerEncoder.build_transformer(config=self.config, training=training)
        model = StructuralAttentionModel(self.config, transformer, self.transformer_tokenizer)
        return model

    def fit(self, trn_data, dev_data, save_dir,
            transformer=None,
            mask_prob=0.15,
            projection=None,
            average_subwords=False,
            transformer_hidden_dropout=None,
            layer_dropout=0,
            mix_embedding: int = None,
            embed_dropout=.33,
            n_mlp_arc=500,
            n_mlp_rel=100,
            mlp_dropout=.33,
            lr=2e-3,
            transformer_lr=5e-5,
            mu=.9,
            nu=.9,
            epsilon=1e-12,
            clip=5.0,
            decay=.75,
            decay_steps=5000,
            patience=100,
            sampler='kmeans',
            n_buckets=32,
            batch_max_tokens=5000,
            batch_size=None,
            epochs=50000,
            tree=False,
            punct=False,
            logger=None,
            verbose=True,
            max_sequence_length=512,
            devices: Union[float, int, List[int]] = None,
            transform=None,
            **kwargs):
        return TorchComponent.fit(self, **merge_locals_kwargs(locals(), kwargs))

    def feed_batch(self, batch):
        if self.model.training:
            input_ids = batch['input_ids']
            prefix_mask = batch['prefix_mask']
            batch['gold_input_ids'] = input_ids
            batch['input_ids'], batch['input_ids_mask'] = self.mlm_generator(input_ids, prefix_mask)
        words, feats, lens, puncts = batch.get('token_id', None), batch.get('pos_id', None), batch['sent_length'], \
                                     batch.get('punct_mask', None)
        mask = lengths_to_mask(lens)
        arc_scores, rel_scores, pred_input_ids = self.model(words=words, feats=feats, mask=mask, batch=batch, **batch)
        batch['pred_input_ids'] = pred_input_ids
        # ignore the first token of each sentence
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        if self.model.training:
            mask = mask.clone()
        mask[:, 0] = 0
        return arc_scores, rel_scores, mask, puncts

    def on_config_ready(self, **kwargs):
        super().on_config_ready(**kwargs)
        self.mlm_generator = MaskedTokenGenerator(self.transformer_tokenizer, self.config.mask_prob)

    def compute_loss(self, arc_scores, rel_scores, arcs, rels, mask, criterion, batch=None):
        parse_loss = BiaffineDependencyParser.compute_loss(self, arc_scores, rel_scores, arcs, rels, mask, criterion, batch)
        if self.model.training:
            gold_input_ids = batch['gold_input_ids']
            pred_input_ids = batch['pred_input_ids']
            input_ids_mask = batch['input_ids_mask']
            token_span = batch['token_span']
            gold_input_ids = batch['gold_input_ids'] = gold_input_ids.gather(1, token_span[:, :, 0])
            input_ids_mask = batch['input_ids_mask'] = input_ids_mask.gather(1, token_span[:, :, 0])
            mlm_loss = F.cross_entropy(pred_input_ids[input_ids_mask], gold_input_ids[input_ids_mask])
            loss = parse_loss + mlm_loss
            return loss
        return parse_loss

    def build_tokenizer_transform(self):
        return TransformerSequenceTokenizer(self.transformer_tokenizer, 'token', '', ret_prefix_mask=True,
                                            ret_token_span=True, cls_is_bos=True,
                                            max_seq_length=self.config.get('max_sequence_length',
                                                                           512),
                                            truncate_long_sequences=False)

    def build_metric(self, training=None, **kwargs):
        parse_metric = super().build_metric(**kwargs)
        if training:
            mlm_metric = CategoricalAccuracy()
            return parse_metric, mlm_metric
        return parse_metric

    def update_metric(self, arc_scores, rel_scores, arcs, rels, mask, puncts, metric, batch=None):
        if isinstance(metric, tuple):
            parse_metric, mlm_metric = metric
            super().update_metric(arc_scores, rel_scores, arcs, rels, mask, puncts, parse_metric)
            gold_input_ids = batch['gold_input_ids']
            input_ids_mask = batch['input_ids_mask']
            pred_input_ids = batch['pred_input_ids']
            pred_input_ids = pred_input_ids[input_ids_mask]
            gold_input_ids = gold_input_ids[input_ids_mask]
            if len(pred_input_ids):
                mlm_metric(pred_input_ids, gold_input_ids)
        else:
            super().update_metric(arc_scores, rel_scores, arcs, rels, mask, puncts, metric)

    def _report(self, loss, metric):
        if isinstance(metric, tuple):
            parse_metric, mlm_metric = metric
            return super()._report(loss, parse_metric) + f' {mlm_metric}'
        else:
            return super()._report(loss, metric)
