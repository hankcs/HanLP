# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-15 14:21

from typing import Dict, Any

import torch

from hanlp.components.parsers.biaffine.biaffine_dep import BiaffineDependencyParser
from hanlp.components.parsers.biaffine.biaffine_model import BiaffineDecoder
from hanlp.components.parsers.ud.tag_decoder import TagDecoder
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbeddingModule
from hanlp.layers.scalar_mix import ScalarMixWithDropout


class UniversalDependenciesModel(torch.nn.Module):
    def __init__(self,
                 encoder: ContextualWordEmbeddingModule,
                 n_mlp_arc,
                 n_mlp_rel,
                 mlp_dropout,
                 num_rels,
                 num_lemmas,
                 num_upos,
                 num_feats,
                 mix_embedding: int = 13,
                 layer_dropout: int = 0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = UniversalDependenciesDecoder(
            encoder.get_output_dim(),
            n_mlp_arc,
            n_mlp_rel,
            mlp_dropout,
            num_rels,
            num_lemmas,
            num_upos,
            num_feats,
            mix_embedding,
            layer_dropout
        )

    def forward(self,
                batch: Dict[str, torch.Tensor],
                mask,
                ):
        hidden = self.encoder(batch)
        return self.decoder(hidden, batch=batch, mask=mask)


class UniversalDependenciesDecoder(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 n_mlp_arc,
                 n_mlp_rel,
                 mlp_dropout,
                 num_rels,
                 num_lemmas,
                 num_upos,
                 num_feats,
                 mix_embedding: int = 13,
                 layer_dropout: int = 0.0,
                 ) -> None:
        super(UniversalDependenciesDecoder, self).__init__()

        # decoders
        self.decoders = torch.nn.ModuleDict({
            'lemmas': TagDecoder(hidden_size, num_lemmas, label_smoothing=0.03, adaptive=True),
            'upos': TagDecoder(hidden_size, num_upos, label_smoothing=0.03, adaptive=True),
            'deps': BiaffineDecoder(hidden_size, n_mlp_arc, n_mlp_rel, mlp_dropout, num_rels),
            'feats': TagDecoder(hidden_size, num_feats, label_smoothing=0.03, adaptive=True),
        })
        self.gold_keys = {
            'lemmas': 'lemma_id',
            'upos': 'pos_id',
            'feats': 'feat_id',
        }

        if mix_embedding:
            self.scalar_mix = torch.nn.ModuleDict({
                task: ScalarMixWithDropout((1, mix_embedding),
                                           do_layer_norm=False,
                                           dropout=layer_dropout)
                for task in self.decoders
            })
        else:
            self.scalar_mix = None

    def forward(self,
                hidden,
                batch: Dict[str, torch.Tensor],
                mask) -> Dict[str, Any]:
        mask_without_root = mask.clone()
        mask_without_root[:, 0] = False

        logits = {}
        class_probabilities = {}
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities}
        loss = 0

        arc = batch.get('arc', None)
        # Run through each of the tasks on the shared encoder and save predictions
        for task in self.decoders:
            if self.scalar_mix:
                decoder_input = self.scalar_mix[task](hidden, mask)
            else:
                decoder_input = hidden

            if task == "deps":
                s_arc, s_rel = self.decoders[task](decoder_input, mask)
                pred_output = {'class_probabilities': {'s_arc': s_arc, 's_rel': s_rel}}
                if arc is not None:
                    # noinspection PyTypeChecker
                    pred_output['loss'] = BiaffineDependencyParser.compute_loss(None, s_arc, s_rel, arc,
                                                                                batch['rel_id'],
                                                                                mask_without_root,
                                                                                torch.nn.functional.cross_entropy)
            else:
                pred_output = self.decoders[task](decoder_input, mask_without_root,
                                                  batch.get(self.gold_keys[task], None))
            if 'logits' in pred_output:
                logits[task] = pred_output["logits"]
            if 'class_probabilities' in pred_output:
                class_probabilities[task] = pred_output["class_probabilities"]
            if 'loss' in pred_output:
                # Keep track of the loss if we have the gold tags available
                loss += pred_output["loss"]

        if arc is not None:
            output_dict["loss"] = loss

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for task in self.tasks:
            self.decoders[task].decode(output_dict)

        return output_dict
