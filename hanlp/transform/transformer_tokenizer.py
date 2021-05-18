# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-03 16:23
import warnings
from typing import Union, Optional

from hanlp_common.constant import BOS, EOS
from hanlp_common.structure import SerializableDict
from hanlp.layers.transformers.pt_imports import PreTrainedTokenizer, PretrainedConfig, AutoTokenizer
from hanlp_trie import DictInterface


class TransformerTokenizer(object):

    def __init__(self, max_seq_length=512, truncate_long_sequences=True) -> None:
        self.truncate_long_sequences = truncate_long_sequences
        self.max_seq_length = max_seq_length

    def sliding_window(self, flat_wordpiece_ids, same_tail=True):
        if same_tail:
            start_piece_ids, flat_wordpiece_ids, end_piece_ids = flat_wordpiece_ids[:1], \
                                                                 flat_wordpiece_ids[1:-1], flat_wordpiece_ids[-1:]
        else:
            start_piece_ids, flat_wordpiece_ids, end_piece_ids = flat_wordpiece_ids[:1], \
                                                                 flat_wordpiece_ids[1:], []
        window_length = self.max_seq_length - len(start_piece_ids) - len(end_piece_ids)
        stride = window_length // 2
        wordpiece_windows = [start_piece_ids + flat_wordpiece_ids[i:i + window_length] + end_piece_ids
                             for i in range(0, len(flat_wordpiece_ids), stride)]

        # Check for overlap in the last window. Throw it away if it is redundant.
        last_window = wordpiece_windows[-1][1:]
        penultimate_window = wordpiece_windows[-2]
        if last_window == penultimate_window[-len(last_window):]:
            wordpiece_windows = wordpiece_windows[:-1]

        wordpiece_ids = [wordpiece for sequence in wordpiece_windows for wordpiece in sequence]
        return wordpiece_ids


class TransformerTextTokenizer(TransformerTokenizer):
    _KEY = ['input_ids', 'attention_mask', 'token_type_ids']

    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, str],
                 text_a_key: str,
                 text_b_key: str = None,
                 output_key=None,
                 max_seq_length=512, truncate_long_sequences=True) -> None:
        super().__init__(max_seq_length, truncate_long_sequences)
        self.text_b = text_b_key
        self.text_a = text_a_key
        if output_key is None:
            output_key = self.text_a
            if text_b_key:
                output_key += '_' + text_b_key
        if output_key == '':
            output_key = self._KEY
        else:
            output_key = [f'{output_key}_{key}' for key in self._KEY]
        self.output_key = output_key
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer

    def __call__(self, sample: dict):
        text_a = sample[self.text_a]
        text_b = sample[self.text_b] if self.text_b else None
        max_seq_length = self.max_seq_length if self.truncate_long_sequences else None
        encoding = self.tokenizer.encode_plus(text_a, text_b, max_length=max_seq_length)
        input_ids = encoding.data['input_ids']
        if not self.truncate_long_sequences and len(input_ids) > self.max_seq_length:
            input_ids = self.sliding_window(input_ids)
            encoding.data['input_ids'] = input_ids  # TODO: other fields should be properly handled too
        for k, v in zip(self.output_key, [encoding.data[_] for _ in self._KEY]):
            sample[k] = v
        return sample


class TransformerSequenceTokenizer(TransformerTokenizer):

    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, str],
                 input_key,
                 output_key=None,
                 max_seq_length=512,
                 truncate_long_sequences=False,
                 config: PretrainedConfig = None,
                 cls_token_at_end=False,
                 cls_token_segment_id=0,
                 pad_token_segment_id=0,
                 pad_on_left=False,
                 do_padding=False,
                 sep_token_extra=False,
                 ret_mask_and_type=False,
                 ret_prefix_mask=False,
                 ret_token_span=True,
                 ret_subtokens=False,
                 ret_subtokens_group=False,
                 cls_is_bos=False,
                 sep_is_eos=False,
                 do_basic_tokenize=True,
                 use_fast=True,
                 dict_force=None,
                 strip_cls_sep=True,
                 check_space_before=None,
                 ) -> None:
        """A transformer tokenizer for token-level tasks. It honors the boundary of tokens and tokenize each token into
        several subtokens then merge them. The information about each subtoken belongs to which token are kept and
        returned as a new field in the sample. It also provides out-of-box sliding window trick on long sequences.

        Args:
            tokenizer: The identifier of a pre-trained tokenizer or a ``PreTrainedTokenizer``.
            input_key: The token key in samples.
            output_key: The output keys to store results.
                max_seq_length: Sentences longer than ``max_seq_len`` will be split into shorter ones if possible.
            truncate_long_sequences: ``True`` to truncate exceeded parts of long sequences. ``False`` to  enable
                sliding window.
            config: The ``PretrainedConfig`` to determine the model structure of the transformer, so that special
                tokenization can be applied.
            cls_token_at_end: ``True`` to put ``[CLS]`` at the end of input tokens.
            cls_token_segment_id: The id of ``[CLS]``.
            pad_token_segment_id: The id of ``[SEP]``.
            pad_on_left: ``True`` to put ``[PAD]`` at the left side of input tokens.
            do_padding: ``True`` to pad sequence to the left.
            sep_token_extra: ``True`` to have two ``[SEP]``.
            ret_mask_and_type: ``True`` to return masks and type ids.
            ret_prefix_mask: ``True`` to generate a mask where each non-zero element corresponds to a prefix of a token.
            ret_token_span: ``True`` to return span of each token measured by subtoken offsets.
            ret_subtokens: ``True`` to return list of subtokens belonging to each token for tokenization purpose.
                When enabled, the prefix mask for each subtoken is set to True as each subtoken is a token unit in
                tokenization task. Similarity, the token span for each token will be a continuous integer sequence.
            ret_subtokens_group: ``True`` to return list of offsets of subtokens belonging to each token.
            cls_is_bos: ``True`` means the first token of input is treated as [CLS] no matter what its surface form is.
                        ``False`` (default) means the first token is not [CLS], it will have its own embedding other than
                        the embedding of [CLS].
            sep_is_eos: ``True`` means the last token of input is [SEP].
                        ``False`` means it's not but [SEP] will be appended,
                        ``None`` means it dependents on `input[-1] == [EOS]`.
            do_basic_tokenize: Whether to do basic tokenization before wordpiece.
            use_fast: Whether or not to try to load the fast version of the tokenizer.
            dict_force: A dictionary doing longest-prefix-match on input text so that the head and tail of each keyword
                won't be concatenated to other tokens by transformer tokenizers.
            strip_cls_sep: ``True`` to strip [CLS] and [SEP] off the input tokens.
            check_space_before: ``True`` to detect the space before each token to handle underline in sentence piece
                tokenization.

        Examples:

        .. highlight:: python
        .. code-block:: python

            transform = TransformerSequenceTokenizer('bert-base-uncased', 'token')
            sample = {'token': 'HanLP good'.split()}
            print(transform(sample))

        """
        super().__init__(max_seq_length, truncate_long_sequences)
        tokenizer_name = tokenizer if isinstance(tokenizer, str) else tokenizer.name_or_path
        if check_space_before is None:
            # These tokenizer is BPE-based which appends a space before each token and tokenizes loving into
            # ['▁lo', 'ving'], tokenize 商品 into ['▁', '商品']. For the later case, the prefix '▁' has to be removed
            # as there is no space between some languages like Chinese
            check_space_before = tokenizer_name in ('xlm-roberta-base', 'xlm-roberta-large', 'google/mt5-small',
                                                    'google/mt5-base')
        self.check_space_before = check_space_before
        self.ret_subtokens_group = ret_subtokens_group
        self.ret_subtokens = ret_subtokens
        self.sep_is_eos = sep_is_eos
        self.ret_prefix_mask = ret_prefix_mask
        self.ret_mask_and_type = ret_mask_and_type
        self.cls_is_bos = cls_is_bos
        self.ret_token_span = ret_token_span
        if not output_key or isinstance(output_key, str):
            suffixes = ['input_ids']
            if ret_mask_and_type:
                suffixes += 'attention_mask', 'token_type_ids'
            if ret_prefix_mask:
                suffixes += ['prefix_mask']
            if ret_token_span:
                suffixes.append('token_span')
            if output_key is None:
                output_key = [f'{input_key}_{key}' for key in suffixes]
            elif output_key == '':
                output_key = suffixes
            else:
                output_key = [f'{output_key}_{key}' for key in suffixes]

        self.input_key = input_key
        self.output_key = output_key
        if config:
            xlnet = config_is(config, 'xlnet')
            pad_token_segment_id = 4 if xlnet else 0
            cls_token_segment_id = 2 if xlnet else 0
            cls_token_at_end = xlnet
            pad_on_left = xlnet
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=use_fast, do_basic_tokenize=do_basic_tokenize)
        if use_fast:
            # Dirty fix upstream bug: https://github.com/hankcs/HanLP/issues/1602
            if hasattr(tokenizer, '_tokenizer') and hasattr(tokenizer._tokenizer, 'no_truncation'):
                _t = tokenizer._tokenizer
                _t.no_truncation()
                _t.no_padding()
                _t.no_truncation = _t.no_padding = lambda: None
        pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
        self.pad_token_segment_id = pad_token_segment_id
        if tokenizer_name in ('google/mt5-small', 'google/mt5-base'):
            # mt5 doesn't have cls or sep, but we can use something similar
            self.has_cls = False
            self.cls_token = '▁'
            self.cls_token_id = tokenizer.convert_tokens_to_ids(self.cls_token)
            self.sep_token = tokenizer.eos_token
            self.sep_token_id = tokenizer.eos_token_id
        else:
            self.has_cls = True
            self.cls_token = tokenizer.cls_token
            self.sep_token = tokenizer.sep_token
            self.cls_token_segment_id = cls_token_segment_id
            self.cls_token_id = tokenizer.cls_token_id
            self.sep_token_id = tokenizer.sep_token_id

        self.sep_token_extra = sep_token_extra
        self.cls_token_at_end = cls_token_at_end
        self.tokenizer = tokenizer
        self.pad_on_left = pad_on_left
        self.do_padding = do_padding
        if self.ret_token_span or not self.truncate_long_sequences:
            assert not self.cls_token_at_end
            assert not self.pad_on_left
        # if self.ret_subtokens:
        #     if not use_fast:
        #         raise NotImplementedError(
        #             'ret_subtokens is not available when using Python tokenizers. '
        #             'To use this feature, set use_fast = True.')
        self.dict: Optional[DictInterface] = dict_force  # For tokenization of raw text
        self.strip_cls_sep = strip_cls_sep

    def __call__(self, sample: dict):
        input_tokens = sample[self.input_key]
        input_is_str = isinstance(input_tokens, str)
        tokenizer = self.tokenizer
        ret_token_span = self.ret_token_span
        if input_is_str:  # This happens in a tokenizer component where the raw sentence is fed.

            # noinspection PyShadowingNames
            def tokenize_str(input_str, add_special_tokens=True):
                if tokenizer.is_fast:
                    encoding = tokenizer.encode_plus(input_str,
                                                     return_offsets_mapping=True,
                                                     add_special_tokens=add_special_tokens).encodings[0]
                    subtoken_offsets = encoding.offsets
                    input_tokens = encoding.tokens
                    input_ids = encoding.ids

                    # Fill up missing non-blank characters swallowed by HF tokenizer
                    offset = 0
                    fixed_offsets = []
                    fixed_tokens = []
                    fixed_ids = []
                    for token, id, (b, e) in zip(input_tokens, input_ids, subtoken_offsets):
                        if b > offset:
                            missing_token = input_str[offset: b]
                            if not missing_token.isspace():  # In the future, we may want space back
                                fixed_tokens.append(missing_token)
                                fixed_ids.append(tokenizer.unk_token_id)
                                fixed_offsets.append((offset, b))
                        fixed_tokens.append(token)
                        fixed_ids.append(id)
                        fixed_offsets.append((b, e))
                        offset = e
                    subtoken_offsets = fixed_offsets
                    input_tokens = fixed_tokens
                    input_ids = fixed_ids

                    if add_special_tokens:
                        subtoken_offsets = subtoken_offsets[1 if self.has_cls else 0:-1]

                    if not self.has_cls:
                        input_tokens = [self.cls_token] + input_tokens
                        input_ids = [self.cls_token_id] + input_ids
                else:
                    input_tokens = tokenizer.tokenize(input_str)
                    subtoken_offsets = []
                    _o = 0
                    for each in input_tokens:
                        subtoken_offsets.append((_o, _o + len(each)))
                        _o += len(each)
                    if add_special_tokens:
                        input_tokens = [self.cls_token] + input_tokens + [self.sep_token]
                    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                if self.check_space_before:
                    non_blank_offsets = [i for i in range(len(input_tokens)) if input_tokens[i] != '▁']
                    if add_special_tokens and not self.has_cls:
                        non_blank_offsets.insert(0, 0)
                    input_tokens = [input_tokens[i] for i in non_blank_offsets]
                    input_ids = [input_ids[i] for i in non_blank_offsets]
                    if add_special_tokens:
                        non_blank_offsets = non_blank_offsets[1:-1]
                        subtoken_offsets = [subtoken_offsets[i - 1] for i in non_blank_offsets]
                    else:
                        subtoken_offsets = [subtoken_offsets[i] for i in non_blank_offsets]
                    # MT5 generates tokens like ▁of, which is bad for the tokenizer. So we want to remove the prefix.
                    for i, token in enumerate(input_tokens[1:-1] if add_special_tokens else input_tokens):
                        if input_str[subtoken_offsets[i][0]] == ' ':
                            subtoken_offsets[i] = (subtoken_offsets[i][0] + 1, subtoken_offsets[i][1])
                if not input_ids:  # This chunk might be some control chars getting removed by tokenizer
                    input_tokens = [input_str]
                    input_ids = [tokenizer.unk_token_id]
                    subtoken_offsets = [(0, len(input_str))]
                return input_tokens, input_ids, subtoken_offsets

            if self.dict:
                chunks = self.dict.split(input_tokens)
                _input_tokens, _input_ids, _subtoken_offsets = [self.cls_token], [self.cls_token_id], []
                _offset = 0
                custom_words = sample['custom_words'] = []
                for chunk in chunks:
                    if isinstance(chunk, str):
                        tokens, ids, offsets = tokenize_str(chunk, add_special_tokens=False)
                    else:
                        begin, end, label = chunk
                        # chunk offset is in char level
                        # custom_words.append(chunk)
                        if isinstance(label, list):
                            tokens, ids, offsets, delta = [], [], [], 0
                            for token in label:
                                _tokens, _ids, _offsets = tokenize_str(token, add_special_tokens=False)
                                tokens.extend(_tokens)
                                # track the subword offset of this chunk, -1 for [CLS]
                                custom_words.append(
                                    (len(_input_ids) + len(ids) - 1, len(_input_ids) + len(ids) - 1 + len(_ids), token))
                                ids.extend(_ids)
                                offsets.extend((x[0] + delta, x[1] + delta) for x in _offsets)
                                delta = offsets[-1][-1]
                        else:
                            tokens, ids, offsets = tokenize_str(input_tokens[begin:end], add_special_tokens=False)
                            # offsets = [(offsets[0][0], offsets[-1][-1])]
                            custom_words.append((len(_input_ids) - 1, len(_input_ids) + len(ids) - 1, label))
                    _input_tokens.extend(tokens)
                    _input_ids.extend(ids)
                    _subtoken_offsets.extend((x[0] + _offset, x[1] + _offset) for x in offsets)
                    _offset = _subtoken_offsets[-1][-1]
                subtoken_offsets = _subtoken_offsets
                input_tokens = _input_tokens + [self.sep_token]
                input_ids = _input_ids + [self.sep_token_id]
            else:
                input_tokens, input_ids, subtoken_offsets = tokenize_str(input_tokens, add_special_tokens=True)

            if self.ret_subtokens:
                sample[f'{self.input_key}_subtoken_offsets'] = subtoken_offsets

        cls_is_bos = self.cls_is_bos
        if cls_is_bos is None:
            cls_is_bos = input_tokens[0] == BOS
        sep_is_eos = self.sep_is_eos
        if sep_is_eos is None:
            sep_is_eos = input_tokens[-1] == EOS
        if self.strip_cls_sep:
            if cls_is_bos:
                input_tokens = input_tokens[1:]
            if sep_is_eos:
                input_tokens = input_tokens[:-1]
        if not self.ret_mask_and_type:  # only need input_ids and token_span, use a light version
            if input_is_str:
                prefix_mask = self._init_prefix_mask(input_ids)
            else:
                if input_tokens:
                    return_offsets_mapping = tokenizer.is_fast and self.ret_subtokens
                    encodings = tokenizer.batch_encode_plus(
                        input_tokens,
                        return_offsets_mapping=return_offsets_mapping,
                        add_special_tokens=False
                    )
                    if return_offsets_mapping:
                        offsets_mapping = [encoding.offsets for encoding in encodings.encodings]
                    else:
                        offsets_mapping = []
                        for token, subtoken_ids in zip(input_tokens, encodings.data['input_ids']):
                            if len(subtoken_ids) > len(token):  # … --> ...
                                del subtoken_ids[len(token):]
                            char_per_subtoken = -(-len(token) // len(subtoken_ids))
                            bes = list(zip(range(0, len(token), char_per_subtoken),
                                           range(char_per_subtoken, len(token) + char_per_subtoken, char_per_subtoken)))
                            if bes[-1][-1] != len(token):
                                bes[-1] = (bes[-1][0], len(token))
                            offsets_mapping.append(bes)
                else:
                    encodings = SerializableDict()
                    encodings.data = {'input_ids': []}
                subtoken_ids_per_token = encodings.data['input_ids']
                if self.check_space_before:
                    # noinspection PyUnboundLocalVariable
                    for token, subtokens, mapping, encoding in zip(input_tokens, subtoken_ids_per_token,
                                                                   offsets_mapping, encodings.encodings):
                        # Remove ▁ generated by spm for 2 reasons:
                        # 1. During decoding, mostly no ▁ will be created unless blanks are placed between tokens (which
                        # is true for English but in English it will likely be concatenated to the token following it)
                        # 2. For T5, '▁' is used as CLS
                        if len(subtokens) > 1 and encoding.tokens[0] == '▁':
                            subtokens.pop(0)
                            if mapping:
                                mapping.pop(0)
                # Some tokens get stripped out
                subtoken_ids_per_token = [ids if ids else [tokenizer.unk_token_id] for ids in subtoken_ids_per_token]
                input_ids = sum(subtoken_ids_per_token, [self.cls_token_id])
                if self.sep_is_eos is None:
                    # None means to check whether sep is at the tail or between tokens
                    if sep_is_eos:
                        input_ids += [self.sep_token_id]
                    elif self.sep_token_id not in input_ids:
                        input_ids += [self.sep_token_id]
                else:
                    input_ids += [self.sep_token_id]
                # else self.sep_is_eos == False means sep is between tokens and don't bother to check

                if self.ret_subtokens:
                    prefix_mask = self._init_prefix_mask(input_ids)
                    # if self.check_space_before:
                    #     if offsets_mapping[0] and not input_tokens[0].startswith(' '):
                    #         prefix_mask[1] = False
                else:
                    prefix_mask = [False] * len(input_ids)
                    offset = 1
                    for _subtokens in subtoken_ids_per_token:
                        prefix_mask[offset] = True
                        offset += len(_subtokens)
                if self.ret_subtokens:
                    subtoken_offsets = []
                    for token, offsets in zip(input_tokens, offsets_mapping):
                        if offsets:
                            subtoken_offsets.append(offsets)
                        else:
                            subtoken_offsets.append([(0, len(token))])
                    if self.ret_subtokens_group:
                        sample[f'{self.input_key}_subtoken_offsets_group'] = subtoken_offsets
                    sample[f'{self.input_key}_subtoken_offsets'] = sum(subtoken_offsets, [])
        else:
            input_ids, attention_mask, token_type_ids, prefix_mask = \
                convert_examples_to_features(input_tokens,
                                             None,
                                             tokenizer,
                                             cls_token_at_end=self.cls_token_at_end,
                                             # xlnet has a cls token at the end
                                             cls_token=tokenizer.cls_token,
                                             cls_token_segment_id=self.cls_token_segment_id,
                                             sep_token=self.sep_token,
                                             sep_token_extra=self.sep_token_extra,
                                             # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                             pad_on_left=self.pad_on_left,
                                             # pad on the left for xlnet
                                             pad_token_id=self.pad_token_id,
                                             pad_token_segment_id=self.pad_token_segment_id,
                                             pad_token_label_id=0,
                                             do_padding=self.do_padding)
        if len(input_ids) > self.max_seq_length:
            if self.truncate_long_sequences:
                # raise SequenceTooLong(
                #     f'Input tokens {input_tokens} exceed the max sequence length of {self.max_seq_length - 2}. '
                #     f'For sequence tasks, truncate_long_sequences = True is not supported.'
                #     f'You are recommended to split your long text into several sentences within '
                #     f'{self.max_seq_length - 2} tokens beforehand. '
                #     f'Or simply set truncate_long_sequences = False to enable sliding window.')
                input_ids = input_ids[:self.max_seq_length]
                prefix_mask = prefix_mask[:self.max_seq_length]
                warnings.warn(
                    f'Input tokens {input_tokens} exceed the max sequence length of {self.max_seq_length - 2}. '
                    f'The exceeded part will be truncated and ignored. '
                    f'You are recommended to split your long text into several sentences within '
                    f'{self.max_seq_length - 2} tokens beforehand.'
                    f'Or simply set truncate_long_sequences = False to enable sliding window.'
                )
            else:
                input_ids = self.sliding_window(input_ids, input_ids[-1] == self.sep_token_id)
        if prefix_mask:
            if cls_is_bos:
                prefix_mask[0] = True
            if sep_is_eos:
                prefix_mask[-1] = True
        outputs = [input_ids]
        if self.ret_mask_and_type:
            # noinspection PyUnboundLocalVariable
            outputs += [attention_mask, token_type_ids]
        if self.ret_prefix_mask:
            outputs += [prefix_mask]
        if ret_token_span and prefix_mask:
            if cls_is_bos:
                token_span = [[0]]
            else:
                token_span = []
            offset = 1
            span = []
            for mask in prefix_mask[1:len(prefix_mask) if sep_is_eos is None else -1]:  # skip [CLS] and [SEP]
                if mask and span:
                    token_span.append(span)
                    span = []
                span.append(offset)
                offset += 1
            if span:
                token_span.append(span)
            if sep_is_eos:
                assert offset == len(prefix_mask) - 1
                token_span.append([offset])
            outputs.append(token_span)
        for k, v in zip(self.output_key, outputs):
            sample[k] = v
        return sample

    def _init_prefix_mask(self, input_ids):
        prefix_mask = [True] * len(input_ids)
        if not self.cls_is_bos:
            prefix_mask[0] = False
        if not self.sep_is_eos:
            prefix_mask[-1] = False
        return prefix_mask


def config_is(config, model='bert'):
    return model in type(config).__name__.lower()


def convert_examples_to_features(
        words,
        max_seq_length: Optional[int],
        tokenizer,
        labels=None,
        label_map=None,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token_id=0,
        pad_token_segment_id=0,
        pad_token_label_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        unk_token='[UNK]',
        do_padding=True
):
    """Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)

    Args:
      words: 
      max_seq_length: 
      tokenizer: 
      labels:  (Default value = None)
      label_map:  (Default value = None)
      cls_token_at_end:  (Default value = False)
      cls_token:  (Default value = "[CLS]")
      cls_token_segment_id:  (Default value = 1)
      sep_token:  (Default value = "[SEP]")
      sep_token_extra:  (Default value = False)
      pad_on_left:  (Default value = False)
      pad_token_id:  (Default value = 0)
      pad_token_segment_id:  (Default value = 0)
      pad_token_label_id:  (Default value = 0)
      sequence_a_segment_id:  (Default value = 0)
      mask_padding_with_zero:  (Default value = True)
      unk_token:  (Default value = '[UNK]')
      do_padding:  (Default value = True)

    Returns:

    """
    args = locals()
    if not labels:
        labels = words
        pad_token_label_id = False

    tokens = []
    label_ids = []
    for word, label in zip(words, labels):
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            # some wired chars cause the tagger to return empty list
            word_tokens = [unk_token] * len(word)
        tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        label_ids.extend([label_map[label] if label_map else True] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    if max_seq_length and len(tokens) > max_seq_length - special_tokens_count:
        warnings.warn(
            f'Input tokens {words} exceed the max sequence length of {max_seq_length - special_tokens_count}. '
            f'The exceeded part will be truncated and ignored. '
            f'You are recommended to split your long text into several sentences within '
            f'{max_seq_length - special_tokens_count} tokens beforehand.')
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  token_type_ids:   0   0   0   0  0     0   0
    #
    # Where "token_type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
        tokens += [cls_token]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    if do_padding:
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token_id] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token_id] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length, f'failed for:\n {args}'
    else:
        assert len(set(len(x) for x in [input_ids, input_mask, segment_ids, label_ids])) == 1
    return input_ids, input_mask, segment_ids, label_ids


def main():
    transformer = 'bert-base-uncased'
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(transformer)
    # _test_text_transform(tokenizer)
    _test_sequence_transform(tokenizer)


def _test_text_transform(tokenizer):
    transform = TransformerTextTokenizer(tokenizer, 'text')
    sample = {'text': 'HanLP good'}
    print(transform(sample))


def _test_sequence_transform(tokenizer):
    transform = TransformerSequenceTokenizer(tokenizer, 'token')
    sample = {'token': 'HanLP good'.split()}
    print(transform(sample))


if __name__ == '__main__':
    main()
