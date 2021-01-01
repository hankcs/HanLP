# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 15:46
from typing import Union

import tensorflow as tf

from hanlp.common.transform_tf import Transform
from hanlp.common.vocab_tf import VocabTF
from hanlp.layers.embeddings.char_cnn_tf import CharCNNEmbeddingTF
from hanlp.layers.embeddings.char_rnn_tf import CharRNNEmbeddingTF
from hanlp.layers.embeddings.concat_embedding import ConcatEmbedding
from hanlp.layers.embeddings.contextual_string_embedding_tf import ContextualStringEmbeddingTF
from hanlp.layers.embeddings.fast_text_tf import FastTextEmbeddingTF
from hanlp.layers.embeddings.word2vec_tf import Word2VecEmbeddingTF, StringWord2VecEmbeddingTF, Word2VecEmbeddingV1

_upgrade = tf.keras.utils.get_custom_objects()
for k, v in list(_upgrade.items()):
    if k.startswith('HanLP>') and k.endswith('TF'):
        _upgrade[k[:-2]] = v


def build_embedding(embeddings: Union[str, int, dict], word_vocab: VocabTF, transform: Transform):
    if not embeddings:
        return None
    config = transform.config
    if isinstance(embeddings, int):
        embeddings = tf.keras.layers.Embedding(input_dim=len(word_vocab), output_dim=embeddings,
                                               trainable=True, mask_zero=True)
        config.embedding_trainable = True
    elif isinstance(embeddings, dict):
        # Upgrade to 2.1
        embed_name = embeddings['class_name'].split('>')[-1]
        if embeddings['class_name'].startswith('HanLP>') and not embeddings['class_name'].endswith('TF'):
            embed_name += 'TF'
        # Embeddings need vocab
        if embed_name in (Word2VecEmbeddingTF.__name__, StringWord2VecEmbeddingTF.__name__):
            # Vocab won't present in the dict
            embeddings['config']['vocab'] = word_vocab
        elif embed_name in (CharRNNEmbeddingTF.__name__, CharCNNEmbeddingTF.__name__):
            embeddings['config']['word_vocab'] = word_vocab
            embeddings['config']['char_vocab'] = transform.char_vocab
            transform.map_x = False
        layer: tf.keras.layers.Embedding = tf.keras.utils.deserialize_keras_object(embeddings)
        # Embedding specific configuration
        if layer.__class__.__name__ in ('FastTextEmbedding', 'FastTextEmbeddingTF'):
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
                if embed['class_name'].split('>')[-1] == Word2VecEmbeddingTF.__name__:
                    embed['class_name'] = 'HanLP>' + StringWord2VecEmbeddingTF.__name__
        return ConcatEmbedding(*[build_embedding(embed, word_vocab, transform) for embed in embeddings])
    else:
        assert isinstance(embeddings, str), 'embedding should be str or int or dict'
        # word_vocab.unlock()
        embeddings = Word2VecEmbeddingV1(path=embeddings, vocab=word_vocab,
                                         trainable=config.get('embedding_trainable', False))
        embeddings = embeddings.array_ks
    return embeddings


def any_embedding_in(embeddings, *cls):
    names = set(x.__name__ for x in cls)
    names.update(list(x[:-2] for x in names if x.endswith('TF')))
    for embed in embeddings:
        if isinstance(embed, dict) and embed['class_name'].split('>')[-1] in names:
            return True
    return False


def embeddings_require_string_input(embeddings):
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
    return any_embedding_in(embeddings, CharRNNEmbeddingTF, CharCNNEmbeddingTF, FastTextEmbeddingTF,
                            ContextualStringEmbeddingTF)


def embeddings_require_char_input(embeddings):
    if not isinstance(embeddings, list):
        embeddings = [embeddings]
    return any_embedding_in(embeddings, CharRNNEmbeddingTF, CharCNNEmbeddingTF, ContextualStringEmbeddingTF)
