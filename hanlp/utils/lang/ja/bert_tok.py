# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-13 13:24
from typing import Union, Optional

from transformers import BertTokenizerFast, TensorType, BatchEncoding, BertJapaneseTokenizer as _BertJapaneseTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput, TruncationStrategy


class BertJapaneseTokenizer(_BertJapaneseTokenizer):
    # We may need to customize character level tokenization to handle English words and URLs
    pass


class BertJapaneseTokenizerFast(BertTokenizerFast):
    def encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        .. warning::
            This method is deprecated, ``__call__`` should be used instead.

        Args:
            text (:obj:`str`, :obj:`List[str]` or :obj:`List[int]` (the latter only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                ``tokenize`` method) or a list of integers (tokenized string ids using the ``convert_tokens_to_ids``
                method).
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the ``tokenize`` method) or a list of integers (tokenized string ids using the
                ``convert_tokens_to_ids`` method).
        """
        text = list(text)
        is_split_into_words = True
        encoding = BertJapaneseTokenizer.encode_plus(self,
                                                     text,
                                                     text_pair,
                                                     add_special_tokens,
                                                     padding,
                                                     truncation,
                                                     max_length,
                                                     stride,
                                                     is_split_into_words,
                                                     pad_to_multiple_of,
                                                     return_tensors,
                                                     return_token_type_ids,
                                                     return_attention_mask,
                                                     return_overflowing_tokens,
                                                     return_special_tokens_mask,
                                                     return_offsets_mapping,
                                                     return_length,
                                                     verbose,
                                                     **kwargs
                                                     )
        offsets = encoding.encodings[0].offsets
        fixed_offsets = [(b + i, e + i) for i, (b, e) in enumerate(offsets)]
        # TODO: This doesn't work with rust tokenizers
        encoding.encodings[0].offsets.clear()
        encoding.encodings[0].offsets.extend(fixed_offsets)
        return encoding
