# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-27 14:30
import logging
from typing import Union, Any, List, Tuple, Iterable

import tensorflow as tf

from hanlp.common.component import KerasComponent
from hanlp.components.taggers.ngram_conv.ngram_conv_tagger import NgramTransform, NgramConvTagger
from hanlp.components.taggers.rnn_tagger import RNNTagger
from hanlp.components.taggers.transformers.transformer_tagger import TransformerTagger
from hanlp.components.taggers.transformers.transformer_transform import TransformerTransform
from hanlp.losses.sparse_categorical_crossentropy import SparseCategoricalCrossentropyOverBatchFirstDim
from hanlp.metrics.chunking.bmes import BMES_F1
from hanlp.transform.tsv import TSVTaggingTransform
from hanlp.transform.txt import extract_ngram_features_and_tags, bmes_to_words, TxtFormat, TxtBMESFormat
from hanlp.utils.util import merge_locals_kwargs


class BMESTokenizer(KerasComponent):

    def build_metrics(self, metrics, logger: logging.Logger, **kwargs):
        if metrics == 'f1':
            self.config.run_eagerly = True
            return BMES_F1(self.transform.tag_vocab)
        return super().build_metrics(metrics, logger, **kwargs)


class NgramConvTokenizerTransform(TxtFormat, NgramTransform):

    def inputs_to_samples(self, inputs, gold=False):
        if self.input_is_single_sample(inputs):
            inputs = [inputs]
        for sent in inputs:
            # bigram_only = false
            yield extract_ngram_features_and_tags(sent, False, self.config.window_size, gold)

    def input_is_single_sample(self, input: Union[List[str], List[List[str]]]) -> bool:
        if not input:
            return True
        return isinstance(input, str)

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None,
                     **kwargs) -> Iterable:
        yield from TxtBMESFormat.Y_to_tokens(self, self.tag_vocab, Y, gold, inputs)


class NgramConvTokenizer(BMESTokenizer, NgramConvTagger):

    def __init__(self) -> None:
        super().__init__(NgramConvTokenizerTransform())

    def fit(self, trn_data: Any, dev_data: Any, save_dir: str, word_embed: Union[str, int, dict] = 200,
            ngram_embed: Union[str, int, dict] = 50, embedding_trainable=True, window_size=4, kernel_size=3,
            filters=(200, 200, 200, 200, 200), dropout_embed=0.2, dropout_hidden=0.2, weight_norm=True,
            loss: Union[tf.keras.losses.Loss, str] = None,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam', metrics='f1', batch_size=100,
            epochs=100, logger=None, verbose=True, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def evaluate_output_to_file(self, batch, outputs, out):
        for x, y_pred in zip(self.transform.X_to_inputs(batch[0]),
                             self.transform.Y_to_outputs(outputs, gold=False)):
            out.write(self.transform.input_truth_output_to_str(x, None, y_pred))
            out.write('\n')

    def build_loss(self, loss, **kwargs):
        if loss is None:
            return SparseCategoricalCrossentropyOverBatchFirstDim()
        return super().build_loss(loss, **kwargs)


class TransformerTokenizerTransform(TxtBMESFormat, TransformerTransform):

    def inputs_to_samples(self, inputs, gold=False):
        yield from TransformerTransform.inputs_to_samples(self, TxtBMESFormat.inputs_to_samples(self, inputs, gold),
                                                          True)

    def Y_to_tokens(self, tag_vocab, Y, gold, inputs):
        if not gold:
            Y = tf.argmax(Y, axis=2)
        for text, ys in zip(inputs, Y):
            tags = [tag_vocab.idx_to_token[int(y)] for y in ys[1:len(text) + 1]]
            yield bmes_to_words(list(text), tags)


class TransformerTokenizer(BMESTokenizer, TransformerTagger):
    def __init__(self, transform: TransformerTokenizerTransform = None) -> None:
        if transform is None:
            transform = TransformerTokenizerTransform()
        super().__init__(transform)


class RNNTokenizerTransform(TxtBMESFormat, TSVTaggingTransform):
    pass


class RNNTokenizer(BMESTokenizer, RNNTagger):
    def __init__(self, transform: RNNTokenizerTransform = None) -> None:
        if not transform:
            transform = RNNTokenizerTransform()
        super().__init__(transform)

    def fit(self, trn_data: str, dev_data: str = None, save_dir: str = None, embeddings=100, embedding_trainable=False,
            rnn_input_dropout=0.2, rnn_units=100, rnn_output_dropout=0.2, epochs=20, lower=False, max_seq_len=50,
            logger=None, loss: Union[tf.keras.losses.Loss, str] = None,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam', metrics='f1', batch_size=32,
            dev_batch_size=32, lr_decay_per_epoch=None, verbose=True, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))
