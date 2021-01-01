# -*- coding: utf-8 -*-
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

from hanlp.components.parsers.biaffine.biaffine import Biaffine
from hanlp.components.parsers.biaffine.mlp import MLP
from hanlp.components.parsers.biaffine.variationalbilstm import VariationalLSTM
from hanlp.layers.dropout import IndependentDropout, SharedDropout, WordDropout
from hanlp.layers.transformers.encoder import TransformerEncoder
from hanlp.layers.transformers.pt_imports import PreTrainedModel, PreTrainedTokenizer
from hanlp.layers.transformers.utils import transformer_encode


class EncoderWithContextualLayer(nn.Module):
    def __init__(self,
                 config,
                 pretrained_embed: torch.Tensor = None,
                 transformer: PreTrainedModel = None,
                 transformer_tokenizer: PreTrainedTokenizer = None,
                 ):
        super(EncoderWithContextualLayer, self).__init__()

        self.secondary_encoder = config.get('secondary_encoder', None)
        self.config = config

        if not transformer:
            self.pad_index = config.pad_index
            self.unk_index = config.unk_index
            if config.word_dropout:
                oov = self.unk_index
                excludes = [self.pad_index]
                self.word_dropout = WordDropout(p=config.word_dropout, oov_token=oov, exclude_tokens=excludes)
            else:
                self.word_dropout = None
        if transformer:
            input_size = 0
            if self.config.transformer_lr:
                hidden_size = transformer.config.hidden_size
            else:
                input_size = transformer.config.hidden_size
                hidden_size = config.n_lstm_hidden * 2
            if config.feat == 'pos':
                self.feat_embed = nn.Embedding(num_embeddings=config.n_feats,
                                               embedding_dim=config.n_embed)
                self.embed_dropout = IndependentDropout(p=config.embed_dropout)
                if self.config.transformer_lr:
                    hidden_size += config.n_embed
                else:
                    input_size += config.n_embed
            if not self.config.transformer_lr:
                self.lstm = VariationalLSTM(input_size=input_size,
                                            hidden_size=config.n_lstm_hidden,
                                            num_layers=config.n_lstm_layers,
                                            dropout=config.hidden_dropout, bidirectional=True)
        else:
            # the embedding layer
            input_size = config.n_embed
            self.word_embed = nn.Embedding(num_embeddings=config.n_words,
                                           embedding_dim=config.n_embed)
            if pretrained_embed is not None:
                if not isinstance(pretrained_embed, torch.Tensor):
                    pretrained_embed = torch.Tensor(pretrained_embed)
                self.pretrained = nn.Embedding.from_pretrained(pretrained_embed)
                nn.init.zeros_(self.word_embed.weight)
            if config.feat == 'pos':
                self.feat_embed = nn.Embedding(num_embeddings=config.n_feats,
                                               embedding_dim=config.n_embed)
                self.embed_dropout = IndependentDropout(p=config.embed_dropout)
                input_size += config.n_embed

            # the word-lstm layer
            hidden_size = config.n_lstm_hidden * 2
            self.lstm = VariationalLSTM(input_size=input_size,
                                        hidden_size=config.n_lstm_hidden,
                                        num_layers=config.n_lstm_layers,
                                        dropout=config.hidden_dropout, bidirectional=True)
        self.hidden_size = hidden_size
        self.hidden_dropout = SharedDropout(p=config.hidden_dropout)
        if transformer:
            transformer = TransformerEncoder(transformer, transformer_tokenizer, config.average_subwords,
                                             word_dropout=config.word_dropout,
                                             max_sequence_length=config.max_sequence_length)
        self.transformer = transformer

    def forward(self, words, feats, input_ids, token_span, mask, lens):
        if mask is None:
            # get the mask and lengths of given batch
            mask = words.ne(self.pad_index)
        if lens is None:
            lens = mask.sum(dim=1)
        batch_size, seq_len = mask.shape
        if self.config.transformer:
            # trans_embed = self.run_transformer(input_ids, token_span=token_span)
            trans_embed = self.transformer.forward(input_ids, token_span=token_span)
            if hasattr(self, 'feat_embed'):
                feat_embed = self.feat_embed(feats)
                trans_embed, feat_embed = self.embed_dropout(trans_embed, feat_embed)
                embed = torch.cat((trans_embed, feat_embed), dim=-1)
            else:
                embed = trans_embed
            if hasattr(self, 'lstm'):
                x = self.run_rnn(embed, lens, seq_len)
            else:
                x = embed
            if self.secondary_encoder:
                x = self.secondary_encoder(x, mask)
            x = self.hidden_dropout(x)
        else:
            if self.word_dropout:
                words = self.word_dropout(words)
            # set the indices larger than num_embeddings to unk_index
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

            # get outputs from embedding layers
            word_embed = self.word_embed(ext_words)
            if hasattr(self, 'pretrained'):
                word_embed += self.pretrained(words)
            if self.config.feat == 'char':
                feat_embed = self.feat_embed(feats[mask])
                feat_embed = pad_sequence(feat_embed.split(lens.tolist()), True)
            elif self.config.feat == 'bert':
                feat_embed = self.feat_embed(*feats)
            elif hasattr(self, 'feat_embed'):
                feat_embed = self.feat_embed(feats)
            else:
                feat_embed = None
            if feat_embed is not None:
                word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
                # concatenate the word and feat representations
                embed = torch.cat((word_embed, feat_embed), dim=-1)
            else:
                embed = word_embed

            x = self.run_rnn(embed, lens, seq_len)
            x = self.hidden_dropout(x)
        return x, mask

    def run_rnn(self, embed, lens, seq_len):
        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        return x

    def run_transformer(self, input_ids, token_span):
        return transformer_encode(self.transformer, input_ids, None, None, token_span,
                                  average_subwords=self.config.average_subwords)


class BiaffineDecoder(nn.Module):
    def __init__(self, hidden_size, n_mlp_arc, n_mlp_rel, mlp_dropout, n_rels, arc_dropout=None,
                 rel_dropout=None) -> None:
        super().__init__()
        # the MLP layers
        self.mlp_arc_h = MLP(hidden_size,
                             n_mlp_arc,
                             dropout=arc_dropout or mlp_dropout)
        self.mlp_arc_d = MLP(hidden_size,
                             n_mlp_arc,
                             dropout=arc_dropout or mlp_dropout)
        self.mlp_rel_h = MLP(hidden_size,
                             n_mlp_rel,
                             dropout=rel_dropout or mlp_dropout)
        self.mlp_rel_d = MLP(hidden_size,
                             n_mlp_rel,
                             dropout=rel_dropout or mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)

    def forward(self, x, mask=None, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        arc_d, arc_h, rel_d, rel_h = self.apply_mlps(x)

        s_arc, s_rel = self.decode(arc_d, arc_h, rel_d, rel_h, mask, self.arc_attn, self.rel_attn)

        return s_arc, s_rel

    @staticmethod
    def decode(arc_d, arc_h, rel_d, rel_h, mask, arc_attn, rel_attn):
        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        if mask is not None:
            # set the scores that exceed the length of each sentence to -inf
            s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        return s_arc, s_rel

    def apply_mlps(self, x):
        # apply MLPs to the hidden states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)
        return arc_d, arc_h, rel_d, rel_h


class BiaffineDependencyModel(nn.Module):

    def __init__(self, config, pretrained_embed: torch.Tensor = None, transformer: PreTrainedModel = None,
                 transformer_tokenizer: PreTrainedTokenizer = None):
        super().__init__()
        self.encoder = EncoderWithContextualLayer(config, pretrained_embed, transformer, transformer_tokenizer)
        self.biaffine_decoder = BiaffineDecoder(self.encoder.hidden_size,
                                                config.n_mlp_arc,
                                                config.n_mlp_rel,
                                                config.mlp_dropout,
                                                config.n_rels)

    def forward(self,
                words=None,
                feats=None,
                input_ids=None,
                token_span=None,
                mask=None, lens=None, **kwargs):
        x, mask = self.encoder(words, feats, input_ids, token_span, mask, lens)
        s_arc, s_rel = self.biaffine_decoder(x, mask)

        return s_arc, s_rel
