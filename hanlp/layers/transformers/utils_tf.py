# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 15:32
import tensorflow as tf
from hanlp.optimizers.adamw import create_optimizer
from hanlp.utils.log_util import logger


def config_is(config, model='bert'):
    return model in type(config).__name__.lower()


def convert_examples_to_features(
        words,
        max_seq_length,
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
    if len(tokens) > max_seq_length - special_tokens_count:
        logger.warning(
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


def build_adamw_optimizer(config, learning_rate, epsilon, clipnorm, train_steps, use_amp, warmup_steps,
                          weight_decay_rate):
    opt = create_optimizer(init_lr=learning_rate,
                           epsilon=epsilon,
                           weight_decay_rate=weight_decay_rate,
                           clipnorm=clipnorm,
                           num_train_steps=train_steps, num_warmup_steps=warmup_steps)
    # opt = tfa.optimizers.AdamW(learning_rate=3e-5, epsilon=1e-08, weight_decay=0.01)
    # opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
    config.optimizer = tf.keras.utils.serialize_keras_object(opt)
    lr_config = config.optimizer['config']['learning_rate']['config']
    if 'decay_schedule_fn' in lr_config:
        lr_config['decay_schedule_fn'] = dict(
            (k, v) for k, v in lr_config['decay_schedule_fn'].items() if not k.startswith('_'))
    if use_amp:
        # loss scaling is currently required when using mixed precision
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
    return opt


def adjust_tokens_for_transformers(sentence):
    """Adjust tokens for BERT
    See https://github.com/DoodleJZ/HPSG-Neural-Parser/blob/master/src_joint/Zparser.py#L1204

    Args:
      sentence: 

    Returns:

    
    """
    cleaned_words = []
    for word in sentence:
        # word = BERT_TOKEN_MAPPING.get(word, word)
        if word == "n't" and cleaned_words:
            cleaned_words[-1] = cleaned_words[-1] + "n"
            word = "'t"
        cleaned_words.append(word)
    return cleaned_words
