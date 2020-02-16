# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-06 13:57
import functools
from typing import Union, List, Any

import torch
from hanlp_common.constant import UNK
from hanlp.common.transform import TransformList
from hanlp.common.vocab import Vocab
from hanlp.components.parsers.biaffine.biaffine import Biaffine
from hanlp.components.parsers.biaffine.biaffine_model import BiaffineDecoder, \
    EncoderWithContextualLayer
from hanlp.components.parsers.biaffine.biaffine_dep import BiaffineDependencyParser
from hanlp.components.parsers.biaffine.biaffine_sdp import BiaffineSemanticDependencyParser
from hanlp_common.conll import CoNLLUWord, CoNLLSentence
from hanlp.components.parsers.parse_alg import add_secondary_arcs_by_preds
from hanlp.datasets.parsing.conll_dataset import append_bos
from hanlp.datasets.parsing.semeval15 import unpack_deps_to_head_deprel, merge_head_deprel_with_2nd
from hanlp.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs
from transformers import PreTrainedModel, PreTrainedTokenizer


class BiaffineSeparateDecoder(torch.nn.Module):

    def __init__(self, hidden_size, config) -> None:
        super().__init__()
        self.biaffine_decoder = BiaffineDecoder(hidden_size,
                                                config.n_mlp_arc,
                                                config.n_mlp_rel,
                                                config.mlp_dropout,
                                                config.n_rels)
        self.biaffine_decoder_2nd = BiaffineDecoder(hidden_size,
                                                    config.n_mlp_arc,
                                                    config.n_mlp_rel,
                                                    config.mlp_dropout,
                                                    config.n_rels_2nd)

    def forward(self, x, mask):
        return tuple(zip(self.biaffine_decoder(x, mask), self.biaffine_decoder_2nd(x, mask)))


class BiaffineJointDecoder(BiaffineDecoder):
    def __init__(self, hidden_size, config) -> None:
        super().__init__(hidden_size, config.n_mlp_arc, config.n_mlp_rel, config.mlp_dropout, config.n_rels)
        # the Biaffine layers for secondary dep
        self.arc_attn_2nd = Biaffine(n_in=config.n_mlp_arc,
                                     bias_x=True,
                                     bias_y=False)
        self.rel_attn_2nd = Biaffine(n_in=config.n_mlp_rel,
                                     n_out=config.n_rels,
                                     bias_x=True,
                                     bias_y=True)

    def forward(self, x, mask=None, **kwargs: Any):
        arc_d, arc_h, rel_d, rel_h = self.apply_mlps(x)
        s_arc, s_rel = self.decode(arc_d, arc_h, rel_d, rel_h, mask, self.arc_attn, self.rel_attn)
        s_arc_2nd, s_rel_2nd = self.decode(arc_d, arc_h, rel_d, rel_h, mask, self.arc_attn_2nd, self.rel_attn_2nd)
        return (s_arc, s_arc_2nd), (s_rel, s_rel_2nd)


class BiaffineSecondaryModel(torch.nn.Module):

    def __init__(self, config, pretrained_embed: torch.Tensor = None, transformer: PreTrainedModel = None,
                 transformer_tokenizer: PreTrainedTokenizer = None):
        super().__init__()
        self.encoder = EncoderWithContextualLayer(config, pretrained_embed, transformer, transformer_tokenizer)
        self.decoder = BiaffineJointDecoder(self.encoder.hidden_size, config) if config.joint \
            else BiaffineSeparateDecoder(self.encoder.hidden_size, config)

    def forward(self,
                words=None,
                feats=None,
                input_ids=None,
                token_span=None,
                mask=None, lens=None, **kwargs):
        x, mask = self.encoder(words, feats, input_ids, token_span, mask, lens)
        return self.decoder(x, mask)


class BiaffineSecondaryParser(BiaffineDependencyParser):

    def __init__(self) -> None:
        super().__init__()
        self.model: BiaffineSecondaryModel = None

    def build_dataset(self, data, bos_transform=None):
        transform = TransformList(functools.partial(append_bos, pos_key='UPOS'),
                                  functools.partial(unpack_deps_to_head_deprel, pad_rel=self.config.pad_rel,
                                                    arc_key='arc_2nd',
                                                    rel_key='rel_2nd'))
        if self.config.joint:
            transform.append(merge_head_deprel_with_2nd)
        if bos_transform:
            transform.append(bos_transform)
        return super().build_dataset(data, transform)

    def build_criterion(self, **kwargs):
        # noinspection PyCallByClass
        return super().build_criterion(**kwargs), (BiaffineSemanticDependencyParser.build_criterion(self, **kwargs))

    def fit(self, trn_data, dev_data, save_dir, feat=None, n_embed=100, pretrained_embed=None, transformer=None,
            average_subwords=False, word_dropout: float = 0.2, transformer_hidden_dropout=None, layer_dropout=0,
            scalar_mix: int = None, embed_dropout=.33, n_lstm_hidden=400, n_lstm_layers=3, hidden_dropout=.33,
            n_mlp_arc=500, n_mlp_rel=100, mlp_dropout=.33, lr=2e-3, transformer_lr=5e-5, mu=.9, nu=.9, epsilon=1e-12,
            clip=5.0, decay=.75, decay_steps=5000, patience=100, batch_size=None, sampler_builder=None,
            lowercase=False, epochs=50000, tree=False, punct=False, min_freq=2,
            apply_constraint=True, joint=False, no_cycle=False, root=None,
            logger=None,
            verbose=True, unk=UNK, pad_rel=None, max_sequence_length=512, devices: Union[float, int, List[int]] = None,
            transform=None, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_vocabs(self, dataset, logger=None, transformer=None):
        self.vocabs['rel_2nd'] = rel_2nd = Vocab(pad_token=self.config.pad_rel, unk_token=self.config.pad_rel)
        if self.config.joint:
            self.vocabs['rel'] = rel_2nd
        super().build_vocabs(dataset, logger, transformer)
        self.config.n_rels_2nd = len(rel_2nd)

    def create_model(self, pretrained_embed, transformer):
        return BiaffineSecondaryModel(self.config, pretrained_embed, transformer, self.transformer_tokenizer)

    def compute_loss(self, arc_scores, rel_scores, arcs, rels, mask, criterion, batch=None):
        arc_scores_1st, arc_scores_2nd, rel_scores_1st, rel_scores_2nd = self.unpack_scores(arc_scores, rel_scores)
        loss_1st = super().compute_loss(arc_scores_1st, rel_scores_1st, arcs, rels, mask, criterion[0], batch)
        mask = self.compute_mask(arc_scores_2nd, batch, mask)
        # noinspection PyCallByClass
        loss_2st = BiaffineSemanticDependencyParser.compute_loss(self, arc_scores_2nd, rel_scores_2nd,
                                                                 batch['arc_2nd'], batch['rel_2nd_id'], mask,
                                                                 criterion[1], batch)
        return loss_1st + loss_2st

    @staticmethod
    def compute_mask(arc_scores_2nd, batch, mask_1st):
        mask = batch.get('mask_2nd', None)
        if mask is None:
            batch['mask_2nd'] = mask = BiaffineSemanticDependencyParser.convert_to_3d_mask(arc_scores_2nd, mask_1st)
        return mask

    def unpack_scores(self, arc_scores, rel_scores):
        arc_scores_1st, arc_scores_2nd = arc_scores
        rel_scores_1st, rel_scores_2nd = rel_scores
        return arc_scores_1st, arc_scores_2nd, rel_scores_1st, rel_scores_2nd

    def get_pad_dict(self):
        d = super(BiaffineSecondaryParser, self).get_pad_dict()
        d.update({'arc_2nd': False})
        return d

    def decode(self, arc_scores, rel_scores, mask, batch=None, predicting=None):
        output_1st, output_2nd = batch.get('outputs', (None, None))
        if output_1st is None:
            arc_scores_1st, arc_scores_2nd, rel_scores_1st, rel_scores_2nd = self.unpack_scores(arc_scores, rel_scores)
            output_1st = super().decode(arc_scores_1st, rel_scores_1st, mask)
            mask = self.compute_mask(arc_scores_2nd, batch, mask)
            # noinspection PyCallByClass
            output_2nd = BiaffineSemanticDependencyParser.decode(self, arc_scores_2nd, rel_scores_2nd, mask, batch)
            if self.config.get('no_cycle'):
                assert predicting, 'No cycle constraint for evaluation is not implemented yet. If you are ' \
                                   'interested, welcome to submit a pull request.'
                root_rel_idx = self.vocabs['rel'].token_to_idx.get(self.config.get('root', None), None)
                arc_pred_1st, rel_pred_1st, arc_pred_2nd, rel_pred_2nd = *output_1st, *output_2nd
                arc_scores_2nd = arc_scores_2nd.transpose(1, 2).cpu().detach().numpy()
                arc_pred_2nd = arc_pred_2nd.cpu().detach().numpy()
                rel_pred_2nd = rel_pred_2nd.cpu().detach().numpy()
                trees = arc_pred_1st.cpu().detach().numpy()
                graphs = []
                for i, (arc_scores, arc_preds, rel_preds, tree, tokens) in enumerate(
                        zip(arc_scores_2nd, arc_pred_2nd, rel_pred_2nd, trees, batch['token'])):
                    sent_len = len(tokens)
                    graph = add_secondary_arcs_by_preds(arc_scores, arc_preds[:sent_len, :sent_len], rel_preds,
                                                        tree[:sent_len], root_rel_idx)
                    graphs.append(graph[1:])  # Remove root
                    # if not predicting:
                    #     # Write back to torch Tensor
                    #     for d, hr in zip(graph):
                    #         pass
                output_2nd = None, graphs

        return tuple(zip(output_1st, output_2nd))

    def update_metric(self, arc_preds, rel_preds, arcs, rels, mask, puncts, metric, batch=None):
        super().update_metric(arc_preds[0], rel_preds[0], arcs, rels, mask, puncts, metric['1st'], batch)
        puncts = BiaffineSemanticDependencyParser.convert_to_3d_puncts(puncts, batch['mask_2nd'])
        # noinspection PyCallByClass
        BiaffineSemanticDependencyParser.update_metric(self, arc_preds[1], rel_preds[1], batch['arc_2nd'],
                                                       batch['rel_2nd_id'], batch['mask_2nd'], puncts, metric['2nd'],
                                                       batch)

    def build_metric(self, **kwargs):
        # noinspection PyCallByClass
        return MetricDict({'1st': super().build_metric(**kwargs),
                           '2nd': BiaffineSemanticDependencyParser.build_metric(self, **kwargs)})

    def collect_outputs_extend(self, predictions: list, arc_preds, rel_preds, lens, mask):
        predictions.extend(rel_preds[1])

    def predictions_to_human(self, predictions, outputs, data, use_pos):
        rel_vocab = self.vocabs['rel'].idx_to_token
        for d, graph in zip(data, predictions):
            sent = CoNLLSentence()
            for idx, (cell, hrs) in enumerate(zip(d, graph)):
                if use_pos:
                    token, pos = cell
                else:
                    token, pos = cell, None
                head = hrs[0][0]
                deprel = rel_vocab[hrs[0][1]]
                deps = [(h, rel_vocab[r]) for h, r in hrs[1:]]
                sent.append(CoNLLUWord(idx + 1, token, upos=pos, head=head, deprel=deprel, deps=deps))
            outputs.append(sent)
