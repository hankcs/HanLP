# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-01-04 06:05
import tensorflow as tf
from transformers import TFAutoModel

from hanlp.layers.transformers.pt_imports import AutoTokenizer_, AutoModel_


def build_transformer(transformer, max_seq_length, num_labels, tagging=True, tokenizer_only=False):
    tokenizer = AutoTokenizer_.from_pretrained(transformer)
    if tokenizer_only:
        return tokenizer
    l_bert = TFAutoModel.from_pretrained(transformer)
    l_input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="input_ids")
    l_mask_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="mask_ids")
    l_token_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name="token_type_ids")
    output = l_bert(input_ids=l_input_ids, token_type_ids=l_token_type_ids, attention_mask=l_mask_ids).last_hidden_state
    if not tagging:
        output = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    logits = tf.keras.layers.Dense(num_labels)(output)
    model = tf.keras.Model(inputs=[l_input_ids, l_mask_ids, l_token_type_ids], outputs=logits)
    model.build(input_shape=(None, max_seq_length))
    return model, tokenizer
