# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-05 13:50
from typing import Optional, Union, List, Any, Dict, Tuple

import torch
from torch import nn

from hanlp_common.configurable import AutoConfigurable
from hanlp.layers.embeddings.embedding import Embedding
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.layers.transformers.encoder import TransformerEncoder
from hanlp.layers.transformers.pt_imports import PreTrainedTokenizer, AutoConfig
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer


class ContextualWordEmbeddingModule(TransformerEncoder):
    def __init__(self,
                 field: str,
                 transformer: str,
                 transformer_tokenizer: PreTrainedTokenizer,
                 average_subwords=False,
                 scalar_mix: Union[ScalarMixWithDropoutBuilder, int] = None,
                 word_dropout=None,
                 max_sequence_length=None,
                 ret_raw_hidden_states=False,
                 transformer_args: Dict[str, Any] = None,
                 trainable=True,
                 training=True) -> None:
        """A contextualized word embedding module.

        Args:
            field: The field to work on. Usually some token fields.
            transformer:  An identifier of a ``PreTrainedModel``.
            transformer_tokenizer:
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
        super().__init__(transformer, transformer_tokenizer, average_subwords, scalar_mix, word_dropout,
                         max_sequence_length, ret_raw_hidden_states, transformer_args, trainable,
                         training)
        self.field = field

    # noinspection PyMethodOverriding
    # noinspection PyTypeChecker
    def forward(self, batch: dict, mask=None, **kwargs):
        input_ids: torch.LongTensor = batch[f'{self.field}_input_ids']
        token_span: torch.LongTensor = batch.get(f'{self.field}_token_span', None)
        # input_device = input_ids.device
        # this_device = self.get_device()
        # if input_device != this_device:
        #     input_ids = input_ids.to(this_device)
        #     token_span = token_span.to(this_device)
        # We might want to apply mask here
        output: Union[torch.Tensor, List[torch.Tensor]] = super().forward(input_ids, token_span=token_span, **kwargs)
        # if input_device != this_device:
        #     if isinstance(output, torch.Tensor):
        #         output = output.to(input_device)
        #     else:
        #         output = [x.to(input_device) for x in output]
        return output

    def get_output_dim(self):
        return self.transformer.config.hidden_size

    def get_device(self):
        device: torch.device = next(self.parameters()).device
        return device


class ContextualWordEmbedding(Embedding, AutoConfigurable):
    def __init__(self, field: str,
                 transformer: str,
                 average_subwords=False,
                 scalar_mix: Union[ScalarMixWithDropoutBuilder, int] = None,
                 word_dropout: Optional[Union[float, Tuple[float, str]]] = None,
                 max_sequence_length=None,
                 truncate_long_sequences=False,
                 cls_is_bos=False,
                 sep_is_eos=False,
                 ret_token_span=True,
                 ret_subtokens=False,
                 ret_subtokens_group=False,
                 ret_prefix_mask=False,
                 ret_raw_hidden_states=False,
                 transformer_args: Dict[str, Any] = None,
                 use_fast=True,
                 do_basic_tokenize=True,
                 trainable=True) -> None:
        """A contextual word embedding builder which builds a
        :class:`~hanlp.layers.embeddings.contextual_word_embedding.ContextualWordEmbeddingModule` and a
        :class:`~hanlp.transform.transformer_tokenizer.TransformerSequenceTokenizer`.

        Args:
            field: The field to work on. Usually some token fields.
            transformer:  An identifier of a ``PreTrainedModel``.
            average_subwords: ``True`` to average subword representations.
            scalar_mix: Layer attention.
            word_dropout: Dropout rate of randomly replacing a subword with MASK.
            max_sequence_length: The maximum sequence length. Sequence longer than this will be handled by sliding
                window.
            truncate_long_sequences: ``True`` to return hidden states of each layer.
            cls_is_bos: ``True`` means the first token of input is treated as [CLS] no matter what its surface form is.
                        ``False`` (default) means the first token is not [CLS], it will have its own embedding other than
                        the embedding of [CLS].
            sep_is_eos: ``True`` means the last token of input is [SEP].
                        ``False`` means it's not but [SEP] will be appended,
                        ``None`` means it dependents on `input[-1] == [EOS]`.
            ret_token_span: ``True`` to return span of each token measured by subtoken offsets.
            ret_subtokens: ``True`` to return list of subtokens belonging to each token.
            ret_subtokens_group: ``True`` to return list of offsets of subtokens belonging to each token.
            ret_prefix_mask: ``True`` to generate a mask where each non-zero element corresponds to a prefix of a token.
            ret_raw_hidden_states: ``True`` to return hidden states of each layer.
            transformer_args: Extra arguments passed to the transformer.
            use_fast: Whether or not to try to load the fast version of the tokenizer.
            do_basic_tokenize: Whether to do basic tokenization before wordpiece.
            trainable: ``False`` to use static embeddings.
        """
        super().__init__()
        self.truncate_long_sequences = truncate_long_sequences
        self.transformer_args = transformer_args
        self.trainable = trainable
        self.ret_subtokens_group = ret_subtokens_group
        self.ret_subtokens = ret_subtokens
        self.ret_raw_hidden_states = ret_raw_hidden_states
        self.sep_is_eos = sep_is_eos
        self.cls_is_bos = cls_is_bos
        self.max_sequence_length = max_sequence_length
        self.word_dropout = word_dropout
        self.scalar_mix = scalar_mix
        self.average_subwords = average_subwords
        self.transformer = transformer
        self.field = field
        self._transformer_tokenizer = TransformerEncoder.build_transformer_tokenizer(self.transformer,
                                                                                     use_fast=use_fast,
                                                                                     do_basic_tokenize=do_basic_tokenize)
        self._tokenizer_transform = TransformerSequenceTokenizer(self._transformer_tokenizer,
                                                                 field,
                                                                 truncate_long_sequences=truncate_long_sequences,
                                                                 ret_prefix_mask=ret_prefix_mask,
                                                                 ret_token_span=ret_token_span,
                                                                 cls_is_bos=cls_is_bos,
                                                                 sep_is_eos=sep_is_eos,
                                                                 ret_subtokens=ret_subtokens,
                                                                 ret_subtokens_group=ret_subtokens_group,
                                                                 max_seq_length=self.max_sequence_length
                                                                 )

    def transform(self, **kwargs) -> TransformerSequenceTokenizer:
        return self._tokenizer_transform

    def module(self, training=True, **kwargs) -> Optional[nn.Module]:
        return ContextualWordEmbeddingModule(self.field,
                                             self.transformer,
                                             self._transformer_tokenizer,
                                             self.average_subwords,
                                             self.scalar_mix,
                                             self.word_dropout,
                                             self.max_sequence_length,
                                             self.ret_raw_hidden_states,
                                             self.transformer_args,
                                             self.trainable,
                                             training=training)

    def get_output_dim(self):
        config = AutoConfig.from_pretrained(self.transformer)
        return config.hidden_size

    def get_tokenizer(self):
        return self._transformer_tokenizer


def find_transformer(embed: nn.Module):
    if isinstance(embed, ContextualWordEmbeddingModule):
        return embed
    if isinstance(embed, nn.ModuleList):
        for child in embed:
            found = find_transformer(child)
            if found:
                return found
