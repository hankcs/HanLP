# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-11-10 13:19

import math
from typing import Union, Tuple, List, Any, Iterable

import tensorflow as tf

from hanlp.common.component import KerasComponent
from hanlp.common.structure import SerializableDict
from hanlp.layers.transformers import BertConfig, BertTokenizer, TFBertForSequenceClassification
from hanlp.optimizers.adamw import create_optimizer
from hanlp.transform.table import TableTransform
from hanlp.utils.log_util import logger
from hanlp.utils.util import merge_locals_kwargs


class BertTextTransform(TableTransform):

    def __init__(self, config: SerializableDict = None, map_x=False, map_y=True, x_columns=None,
                 y_column=-1, skip_header=True, delimiter='auto', **kwargs) -> None:
        super().__init__(config, map_x, map_y, x_columns, y_column, skip_header, delimiter, **kwargs)
        self.tokenizer: BertTokenizer = None

    def build_config(self):
        super().build_config()
        bert = self.config.bert
        self.tokenizer = BertTokenizer.from_pretrained(bert)

    def inputs_to_samples(self, inputs, gold=False):
        tokenizer = self.tokenizer
        max_length = self.config.max_length
        num_features = None
        for (X, Y) in super().inputs_to_samples(inputs, gold):
            if self.label_vocab.mutable:
                yield None, Y
                continue
            if isinstance(X, str):
                X = (X,)
            if num_features is None:
                num_features = self.config.num_features
            assert num_features == len(X), f'Numbers of features {num_features} ' \
                                           f'inconsistent with current {len(X)}'
            text_a = X[0]
            text_b = X[1] if len(X) > 1 else None
            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            mask_padding_with_zero = True
            pad_on_left = False
            pad_token = 0
            pad_token_segment_id = 0
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                max_length)
            label = Y
            yield (input_ids, attention_mask, token_type_ids), label

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        types = (tf.int32, tf.int32, tf.int32), tf.string
        shapes = ([None], [None], [None]), []
        values = (0, 0, 0), self.label_vocab.safe_pad_token
        return types, shapes, values

    def x_to_idx(self, x) -> Union[tf.Tensor, Tuple]:
        logger.fatal('map_x should always be set to True')
        exit(1)

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None) -> Iterable:
        preds = Y[0]
        preds = tf.argmax(preds, axis=1)
        for y in preds:
            yield self.label_vocab.idx_to_token[y]

    def input_is_single_sample(self, input: Any) -> bool:
        return isinstance(input, (str, tuple))


class BertTextClassifier(KerasComponent):

    def __init__(self, bert_text_transform=None) -> None:
        if not bert_text_transform:
            bert_text_transform = BertTextTransform()
        super().__init__(bert_text_transform)
        self.model: TFBertForSequenceClassification
        self.transform: BertTextTransform

    def classify(self, docs: Union[str, List[str], Tuple[str, str]], batch_size=32) -> Union[str, List[str]]:
        flat = False
        if isinstance(docs, (str, tuple)):
            docs = [docs]
            flat = True
        dataset = self.transform.inputs_to_dataset(docs, batch_size=batch_size, cache=False, gold=False)
        outputs = []
        for batch in dataset:
            preds = self.model.predict_on_batch(batch[0])[0]
            preds = tf.argmax(preds, axis=1)
            for y in preds:
                outputs.append(self._y_id_to_str(y))
        if flat:
            outputs = outputs[0]
        return outputs

    # noinspection PyMethodOverriding
    def fit(self, trn_data: Any, dev_data: Any, save_dir: str, bert: str, max_length: int = 128,
            optimizer='adamw', warmup_steps_ratio=0.1, use_amp=False, batch_size=32,
            epochs=3, logger=None, verbose=1, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def evaluate_output(self, tst_data, out, num_batches, metric):
        out.write('sentence\tpred\tgold\n')
        total, correct, score = 0, 0, 0
        for idx, batch in enumerate(tst_data):
            outputs = self.model.predict_on_batch(batch[0])[0]
            outputs = tf.argmax(outputs, axis=1)
            for X, Y_pred, Y_gold, in zip(batch[0][0], outputs, batch[1]):
                feature = ' '.join(self.transform.tokenizer.convert_ids_to_tokens(X.numpy(), skip_special_tokens=True))
                feature = feature.replace(' ##', '')  # fix sub-word generated by BERT tokenizer
                out.write('{}\t{}\t{}\n'.format(feature,
                                                self._y_id_to_str(Y_pred),
                                                self._y_id_to_str(Y_gold)))
                total += 1
                correct += int(tf.equal(Y_pred, Y_gold).numpy())
            score = correct / total
            print('\r{}/{} {}: {:.2f}'.format(idx + 1, num_batches, metric, score * 100), end='')
        print()
        return score

    def _y_id_to_str(self, Y_pred) -> str:
        return self.transform.label_vocab.idx_to_token[Y_pred.numpy()]

    def build_loss(self, loss, **kwargs):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss

    # noinspection PyMethodOverriding
    def build_optimizer(self, optimizer, use_amp, train_steps, warmup_steps, **kwargs):
        if optimizer == 'adamw':
            opt = create_optimizer(init_lr=5e-5, num_train_steps=train_steps, num_warmup_steps=warmup_steps)
            # opt = tfa.optimizers.AdamW(learning_rate=3e-5, epsilon=1e-08, weight_decay=0.01)
            # opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
            self.config.optimizer = tf.keras.utils.serialize_keras_object(opt)
            lr_config = self.config.optimizer['config']['learning_rate']['config']
            lr_config['decay_schedule_fn'] = dict(
                (k, v) for k, v in lr_config['decay_schedule_fn'].items() if not k.startswith('_'))
        else:
            opt = super().build_optimizer(optimizer)
        if use_amp:
            # loss scaling is currently required when using mixed precision
            opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
        return opt

    # noinspection PyMethodOverriding
    def build_model(self, bert, **kwargs):
        bert_config = BertConfig.from_pretrained(bert, num_labels=len(self.transform.label_vocab))
        return TFBertForSequenceClassification.from_pretrained(bert, config=bert_config)

    def build_vocab(self, trn_data, logger):
        train_examples = super().build_vocab(trn_data, logger)
        warmup_steps_per_epoch = math.ceil(train_examples * self.config.warmup_steps_ratio / self.config.batch_size)
        self.config.warmup_steps = warmup_steps_per_epoch * self.config.epochs
        return train_examples
