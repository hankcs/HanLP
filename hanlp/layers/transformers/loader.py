# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-04 06:05
import tensorflow as tf
from bert import BertModelLayer
from bert.loader import _checkpoint_exists, bert_prefix
from bert.loader_albert import map_to_tfhub_albert_variable_name
from tensorflow import keras


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
