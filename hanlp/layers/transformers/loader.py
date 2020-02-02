# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-04 06:05
import glob
import os

import bert
import tensorflow as tf
from bert import BertModelLayer, albert_models_tfhub, fetch_tfhub_albert_model
from bert.loader import _checkpoint_exists, bert_prefix, bert_models_google
from bert.loader_albert import map_to_tfhub_albert_variable_name, albert_params
from tensorflow import keras

from hanlp.layers.transformers import zh_albert_models_google
from hanlp.utils.io_util import get_resource, stdout_redirected, hanlp_home


def load_stock_weights(bert: BertModelLayer, ckpt_path):
    """
    Use this method to load the weights from a pre-trained BERT checkpoint into a bert layer.

    :param bert: a BertModelLayer instance within a built keras model.
    :param ckpt_path: checkpoint path, i.e. `uncased_L-12_H-768_A-12/bert_model.ckpt` or `albert_base_zh/albert_model.ckpt`
    :return: list of weights with mismatched shapes. This can be used to extend
    the segment/token_type embeddings.
    """
    assert isinstance(bert, BertModelLayer), "Expecting a BertModelLayer instance as first argument"
    assert _checkpoint_exists(ckpt_path), "Checkpoint does not exist: {}".format(ckpt_path)
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)

    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())

    prefix = bert_prefix(bert)

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []

    bert_params = bert.weights
    param_values = keras.backend.batch_get_value(bert.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_to_tfhub_albert_variable_name(param.name, prefix)

        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)

            if param_value.shape != ckpt_value.shape:
                print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_path))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)

    print("Done loading {} BERT weights from: {} into {} (prefix:{}). "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
        len(weight_value_tuples), ckpt_path, bert, prefix, skip_count, len(skipped_weight_value_tuples)))

    print("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(sorted(stock_weights.difference(loaded_weights))))

    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)


def build_transformer(transformer, max_seq_length, num_labels, tagging=True, tokenizer_only=False):
    spm_model_file = None
    if transformer in zh_albert_models_google:
        from bert.tokenization.albert_tokenization import FullTokenizer
        model_url = zh_albert_models_google[transformer]
        albert = True
    elif transformer in albert_models_tfhub:
        from hanlp.layers.transformers.albert_tokenization import FullTokenizer
        with stdout_redirected(to=os.devnull):
            model_url = fetch_tfhub_albert_model(transformer,
                                                 os.path.join(hanlp_home(), 'thirdparty', 'tfhub.dev', 'google',
                                                              transformer))
        albert = True
        spm_model_file = glob.glob(os.path.join(model_url, 'assets', '*.model'))
        assert len(spm_model_file) == 1, 'No vocab found or unambiguous vocabs found'
        spm_model_file = spm_model_file[0]
    elif transformer in bert_models_google:
        from bert.tokenization.bert_tokenization import FullTokenizer
        model_url = bert_models_google[transformer]
        albert = False
    else:
        raise ValueError(
            f'Unknown model {transformer}, available ones: {list(bert_models_google.keys()) + list(zh_albert_models_google.keys()) + list(albert_models_tfhub.keys())}')
    bert_dir = get_resource(model_url)
    if spm_model_file:
        vocab = glob.glob(os.path.join(bert_dir, 'assets', '*.vocab'))
    else:
        vocab = glob.glob(os.path.join(bert_dir, '*vocab*.txt'))
    assert len(vocab) == 1, 'No vocab found or unambiguous vocabs found'
    vocab = vocab[0]
    lower_case = any(key in transformer for key in ['uncased', 'multilingual', 'chinese', 'albert'])
    if spm_model_file:
        # noinspection PyTypeChecker
        tokenizer = FullTokenizer(vocab_file=vocab, spm_model_file=spm_model_file, do_lower_case=lower_case)
    else:
        tokenizer = FullTokenizer(vocab_file=vocab, do_lower_case=lower_case)
    if tokenizer_only:
        return tokenizer
    if spm_model_file:
        bert_params = albert_params(bert_dir)
    else:
        bert_params = bert.params_from_pretrained_ckpt(bert_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name='albert' if albert else "bert")
    l_input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="input_ids")
    l_mask_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="mask_ids")
    l_token_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="token_type_ids")
    output = l_bert([l_input_ids, l_token_type_ids], mask=l_mask_ids)
    if not tagging:
        output = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    if bert_params.hidden_dropout:
        output = tf.keras.layers.Dropout(bert_params.hidden_dropout, name='hidden_dropout')(output)
    logits = tf.keras.layers.Dense(num_labels, kernel_initializer=tf.keras.initializers.TruncatedNormal(
        bert_params.initializer_range))(output)
    model = tf.keras.Model(inputs=[l_input_ids, l_mask_ids, l_token_type_ids], outputs=logits)
    model.build(input_shape=(None, max_seq_length))
    if not spm_model_file:
        ckpt = glob.glob(os.path.join(bert_dir, '*.index'))
        assert ckpt, f'No checkpoint found under {bert_dir}'
        ckpt, _ = os.path.splitext(ckpt[0])
    with stdout_redirected(to=os.devnull):
        if albert:
            if spm_model_file:
                skipped_weight_value_tuples = bert.load_albert_weights(l_bert, bert_dir)
            else:
                # noinspection PyUnboundLocalVariable
                skipped_weight_value_tuples = load_stock_weights(l_bert, ckpt)
        else:
            # noinspection PyUnboundLocalVariable
            skipped_weight_value_tuples = bert.load_bert_weights(l_bert, ckpt)
    assert 0 == len(skipped_weight_value_tuples), f'failed to load pretrained {transformer}'
    return model, tokenizer
