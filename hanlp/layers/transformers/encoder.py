# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-22 21:06
import warnings
from typing import Union, Dict, Any, Sequence, Tuple, Optional

import torch
from torch import nn
from hanlp.layers.dropout import WordDropout
from hanlp.layers.scalar_mix import ScalarMixWithDropout, ScalarMixWithDropoutBuilder
from hanlp.layers.transformers.pt_imports import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModel_, \
    BertTokenizer
from hanlp.layers.transformers.utils import transformer_encode


# noinspection PyAbstractClass
class TransformerEncoder(nn.Module):
    def __init__(self,
                 transformer: Union[PreTrainedModel, str],
                 transformer_tokenizer: PreTrainedTokenizer,
                 average_subwords=False,
                 scalar_mix: Union[ScalarMixWithDropoutBuilder, int] = None,
                 word_dropout=None,
                 max_sequence_length=None,
                 ret_raw_hidden_states=False,
                 transformer_args: Dict[str, Any] = None,
                 trainable=Union[bool, Optional[Tuple[int, int]]],
                 training=True) -> None:
        """A pre-trained transformer encoder.

        Args:
            transformer: A ``PreTrainedModel`` or an identifier of a ``PreTrainedModel``.
            transformer_tokenizer: A ``PreTrainedTokenizer``.
            average_subwords: ``True`` to average subword representations.
            scalar_mix: Layer attention.
            word_dropout: Dropout rate of randomly replacing a subword with MASK.
            max_sequence_length: The maximum sequence length. Sequence longer than this will be handled by sliding
                window.
            ret_raw_hidden_states: ``True`` to return hidden states of each layer.
            transformer_args: Extra arguments passed to the transformer.
            trainable: ``False`` to use static embeddings.
            training: ``False`` to skip loading weights from pre-trained transformers.
        """
        super().__init__()
        self.ret_raw_hidden_states = ret_raw_hidden_states
        self.max_sequence_length = max_sequence_length
        self.average_subwords = average_subwords
        if word_dropout:
            oov = transformer_tokenizer.mask_token_id
            if isinstance(word_dropout, Sequence):
                word_dropout, replacement = word_dropout
                if replacement == 'unk':
                    # Electra English has to use unk
                    oov = transformer_tokenizer.unk_token_id
                elif replacement == 'mask':
                    # UDify uses [MASK]
                    oov = transformer_tokenizer.mask_token_id
                else:
                    oov = replacement
            pad = transformer_tokenizer.pad_token_id
            cls = transformer_tokenizer.cls_token_id
            sep = transformer_tokenizer.sep_token_id
            excludes = [pad, cls, sep]
            self.word_dropout = WordDropout(p=word_dropout, oov_token=oov, exclude_tokens=excludes)
        else:
            self.word_dropout = None
        if isinstance(transformer, str):
            output_hidden_states = scalar_mix is not None
            if transformer_args is None:
                transformer_args = dict()
            transformer_args['output_hidden_states'] = output_hidden_states
            transformer = AutoModel_.from_pretrained(transformer, training=training or not trainable,
                                                     **transformer_args)
        if hasattr(transformer, 'encoder') and hasattr(transformer, 'decoder'):
            # For seq2seq model, use its encoder
            transformer = transformer.encoder
        self.transformer = transformer
        if not trainable:
            transformer.requires_grad_(False)
        elif isinstance(trainable, tuple):
            layers = []
            if hasattr(transformer, 'embeddings'):
                layers.append(transformer.embeddings)
            layers.extend(transformer.encoder.layer)
            for i, layer in enumerate(layers):
                if i < trainable[0] or i >= trainable[1]:
                    layer.requires_grad_(False)

        if isinstance(scalar_mix, ScalarMixWithDropoutBuilder):
            self.scalar_mix: ScalarMixWithDropout = scalar_mix.build()
        else:
            self.scalar_mix = None

    def forward(self, input_ids: torch.LongTensor, attention_mask=None, token_type_ids=None, token_span=None, **kwargs):
        if self.word_dropout:
            input_ids = self.word_dropout(input_ids)

        x = transformer_encode(self.transformer,
                               input_ids,
                               attention_mask,
                               token_type_ids,
                               token_span,
                               layer_range=self.scalar_mix.mixture_range if self.scalar_mix else 0,
                               max_sequence_length=self.max_sequence_length,
                               average_subwords=self.average_subwords,
                               ret_raw_hidden_states=self.ret_raw_hidden_states)
        if self.ret_raw_hidden_states:
            x, raw_hidden_states = x
        if self.scalar_mix:
            x = self.scalar_mix(x)
        if self.ret_raw_hidden_states:
            # noinspection PyUnboundLocalVariable
            return x, raw_hidden_states
        return x

    @staticmethod
    def build_transformer(config, training=True) -> PreTrainedModel:
        kwargs = {}
        if config.scalar_mix and config.scalar_mix > 0:
            kwargs['output_hidden_states'] = True
        transformer = AutoModel_.from_pretrained(config.transformer, training=training, **kwargs)
        return transformer

    @staticmethod
    def build_transformer_tokenizer(config_or_str, use_fast=True, do_basic_tokenize=True) -> PreTrainedTokenizer:
        if isinstance(config_or_str, str):
            transformer = config_or_str
        else:
            transformer = config_or_str.transformer
        if use_fast and not do_basic_tokenize:
            warnings.warn('`do_basic_tokenize=False` might not work when `use_fast=True`')
        additional_config = dict()
        if transformer.startswith('voidful/albert_chinese_'):
            cls = BertTokenizer
        elif transformer == 'cl-tohoku/bert-base-japanese-char':
            # Since it's char level model, it's OK to use char level tok instead of fugashi
            # from hanlp.utils.lang.ja.bert_tok import BertJapaneseTokenizerFast
            # cls = BertJapaneseTokenizerFast
            from transformers import BertJapaneseTokenizer
            cls = BertJapaneseTokenizer
            # from transformers import BertTokenizerFast
            # cls = BertTokenizerFast
            additional_config['word_tokenizer_type'] = 'basic'
        else:
            cls = AutoTokenizer
        return cls.from_pretrained(transformer, use_fast=use_fast, do_basic_tokenize=do_basic_tokenize,
                                   **additional_config)
