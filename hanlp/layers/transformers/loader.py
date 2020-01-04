# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-04 06:05
import glob
import os

import bert
import tensorflow as tf
from bert import BertModelLayer
from bert.loader import _checkpoint_exists, bert_prefix, bert_models_google
from bert.loader_albert import map_to_tfhub_albert_variable_name
from tensorflow import keras

from hanlp.layers.transformers import albert_models_google
from hanlp.utils.io_util import get_resource, stdout_redirected


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


def build_transformer(transformer, max_seq_length, num_labels, tagging=True):
    if transformer in albert_models_google:
        from bert.tokenization.albert_tokenization import FullTokenizer
        model_url = albert_models_google[transformer]
        albert = True
    elif transformer in bert_models_google:
        from bert.tokenization.bert_tokenization import FullTokenizer
        model_url = bert_models_google[transformer]
        albert = False
    else:
        raise ValueError(
            f'Unknown model {transformer}, available ones: {bert_models_google.keys() + albert_models_google.keys()}')
    bert_dir = get_resource(model_url)
    vocab = glob.glob(os.path.join(bert_dir, '*vocab*.txt'))
    assert len(vocab) == 1, 'No vocab found or unambiguous vocabs found'
    vocab = vocab[0]
    # noinspection PyTypeChecker
    tokenizer = FullTokenizer(vocab_file=vocab)
    bert_params = bert.params_from_pretrained_ckpt(bert_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
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
    ckpt = glob.glob(os.path.join(bert_dir, '*.index'))
    assert ckpt, f'No checkpoint found under {bert_dir}'
    ckpt, _ = os.path.splitext(ckpt[0])
    with stdout_redirected(to=os.devnull):
        if albert:
            skipped_weight_value_tuples = load_stock_weights(l_bert, ckpt)
        else:
            skipped_weight_value_tuples = bert.load_bert_weights(l_bert, ckpt)
    assert 0 == len(skipped_weight_value_tuples), f'failed to load pretrained {transformer}'
    return model, tokenizer
