# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-24 21:48
from typing import Union

import tensorflow as tf

from hanlp.common.transform import Transform
from hanlp.common.vocab import Vocab
from hanlp.layers.embeddings.char_cnn import CharCNNEmbedding
from hanlp.layers.embeddings.char_rnn import CharRNNEmbedding
from hanlp.layers.embeddings.concat_embedding import ConcatEmbedding
from hanlp.layers.embeddings.contextual_string_embedding import ContextualStringEmbedding
from hanlp.layers.embeddings.word2vec import Word2VecEmbeddingV1, Word2VecEmbedding, StringWord2VecEmbedding


def build_embedding(embeddings: Union[str, int, dict], word_vocab: Vocab, transform: Transform):
    config = transform.config
    if isinstance(embeddings, int):
        embeddings = tf.keras.layers.Embedding(input_dim=len(word_vocab), output_dim=embeddings,
                                               trainable=True, mask_zero=True)
        config.embedding_trainable = True
    elif isinstance(embeddings, dict):
        # Embeddings need vocab
        if embeddings['class_name'].split('>')[-1] in (Word2VecEmbedding.__name__, StringWord2VecEmbedding.__name__):
            # Vocab won't present in the dict
            embeddings['config']['vocab'] = word_vocab
        elif embeddings['class_name'].split('>')[-1] in (CharRNNEmbedding.__name__, CharCNNEmbedding.__name__):
            embeddings['config']['word_vocab'] = word_vocab
            embeddings['config']['char_vocab'] = transform.char_vocab
            transform.map_x = False
        elif embeddings['class_name'].split('>')[-1] == 'FastTextEmbedding':
            from hanlp.layers.embeddings.fast_text import FastTextEmbedding
        layer: tf.keras.layers.Embedding = tf.keras.utils.deserialize_keras_object(embeddings,
                                                                                   custom_objects=tf.keras.utils.get_custom_objects())
        # Embedding specific configuration
        if layer.__class__.__name__ == 'FastTextEmbedding':
            config.run_eagerly = True  # fasttext can only run in eager mode
            config.embedding_trainable = False
            transform.map_x = False  # fasttext accept string instead of int
        return layer
    elif isinstance(embeddings, list):
        if embeddings_require_string_input(embeddings):
            # those embeddings require string as input
            transform.map_x = False
            # use the string version of Word2VecEmbedding instead
            for embed in embeddings:
                if embed['class_name'].split('>')[-1] == Word2VecEmbedding.__name__:
                    embed['class_name'] = 'HanLP>' + StringWord2VecEmbedding.__name__
        return ConcatEmbedding(*[build_embedding(embed, word_vocab, transform) for embed in embeddings])
    else:
        assert isinstance(embeddings, str), 'embedding should be str or int or dict'
        # word_vocab.unlock()
        embeddings = Word2VecEmbeddingV1(path=embeddings, vocab=word_vocab,
                                         trainable=config.get('embedding_trainable', False))
        embeddings = embeddings.array_ks
    return embeddings


def any_embedding_in(embeddings, *cls):
    names = set(x if isinstance(x, str) else x.__name__ for x in cls)
    for embed in embeddings:
        if isinstance(embed, dict) and embed['class_name'].split('>')[-1] in names:
            return True
    return False


def embeddings_require_string_input(embeddings):
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
    return any_embedding_in(embeddings, CharRNNEmbedding, CharCNNEmbedding, 'FastTextEmbedding',
                            ContextualStringEmbedding)


def embeddings_require_char_input(embeddings):
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
    return any_embedding_in(embeddings, CharRNNEmbedding, CharCNNEmbedding, ContextualStringEmbedding)
