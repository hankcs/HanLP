# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-22 12:47
import logging
import math
import os
from typing import List
import numpy as np
import tensorflow as tf

from hanlp.components.parsers.parse_alg import unique_root, adjust_root_score, chu_liu_edmonds
from hanlp.layers.transformers.loader_tf import build_transformer

from hanlp.common.keras_component import KerasComponent
from hanlp.components.parsers.alg_tf import tarjan
from hanlp.components.parsers.biaffine_tf.model import BiaffineModelTF, StructuralAttentionModel
from hanlp.transform.conll_tf import CoNLL_DEP_Transform, CoNLL_Transformer_Transform, CoNLL_SDP_Transform
from hanlp.layers.embeddings.util_tf import build_embedding
from hanlp.layers.transformers.tf_imports import PreTrainedTokenizer, TFAutoModel, TFPreTrainedModel, AutoTokenizer, \
    TFAutoModelWithLMHead, BertTokenizerFast, AlbertConfig, BertTokenizer, TFBertModel
from hanlp.layers.transformers.utils_tf import build_adamw_optimizer
from hanlp.metrics.parsing.labeled_f1_tf import LabeledF1TF
from hanlp.metrics.parsing.labeled_score import LabeledScore
from hanlp_common.util import merge_locals_kwargs


class BiaffineDependencyParserTF(KerasComponent):
    def __init__(self, transform: CoNLL_DEP_Transform = None) -> None:
        if not transform:
            transform = CoNLL_DEP_Transform()
        super().__init__(transform)
        self.transform: CoNLL_DEP_Transform = transform
        self.model: BiaffineModelTF = None

    def build_model(self, pretrained_embed, n_embed, training, **kwargs) -> tf.keras.Model:
        if training:
            self.config.n_words = len(self.transform.form_vocab)
        else:
            self.config.lstm_dropout = 0.  # keras will use cuda lstm when config.lstm_dropout is 0
        self.config.n_feats = len(self.transform.cpos_vocab)
        self._init_config()
        pretrained: tf.keras.layers.Embedding = build_embedding(pretrained_embed, self.transform.form_vocab,
                                                                self.transform) if pretrained_embed else None
        if pretrained_embed:
            self.config.n_embed = pretrained.output_dim
        model = BiaffineModelTF(self.config, pretrained)
        return model

    def _init_config(self):
        self.config.n_rels = len(self.transform.rel_vocab)
        self.config.pad_index = self.transform.form_vocab.pad_idx
        self.config.unk_index = self.transform.form_vocab.unk_idx
        self.config.bos_index = 2

    def load_weights(self, save_dir, filename='model.h5', functional=False, **kwargs):
        super().load_weights(save_dir, filename)
        if functional:
            self.model = self.model.to_functional()

    def fit(self, trn_data, dev_data, save_dir,
            n_embed=100,
            pretrained_embed=None,
            embed_dropout=.33,
            n_lstm_hidden=400,
            n_lstm_layers=3,
            lstm_dropout=.33,
            n_mlp_arc=500,
            n_mlp_rel=100,
            mlp_dropout=.33,
            optimizer='adam',
            lr=2e-3,
            mu=.9,
            nu=.9,
            epsilon=1e-12,
            clip=5.0,
            decay=.75,
            decay_steps=5000,
            patience=100,
            arc_loss='sparse_categorical_crossentropy',
            rel_loss='sparse_categorical_crossentropy',
            metrics=('UAS', 'LAS'),
            n_buckets=32,
            batch_size=5000,
            epochs=50000,
            early_stopping_patience=100,
            tree=False,
            punct=False,
            min_freq=2,
            run_eagerly=False, logger=None, verbose=True,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    # noinspection PyMethodOverriding
    def train_loop(self, trn_data, dev_data, epochs, num_examples,
                   train_steps_per_epoch, dev_steps, model, optimizer, loss, metrics,
                   callbacks, logger: logging.Logger, arc_loss, rel_loss,
                   **kwargs):
        arc_loss, rel_loss = loss
        # because we are customizing batching
        train_steps_per_epoch = len(list(iter(trn_data)))
        # progbar: tf.keras.callbacks.ProgbarLogger = callbacks[-1]
        c: tf.keras.callbacks.Callback = None
        metric = self._build_metrics()
        for c in callbacks:
            if not hasattr(c, 'params'):
                c.params = dict()
            c.params['epochs'] = epochs
            c.params['trn_data'] = trn_data
            c.params['metrics'] = ['loss'] + self.config.metrics
            c.params['metrics'] = c.params['metrics'] + [f'val_{k}' for k in c.params['metrics']]
            c.on_train_begin()
        for epoch in range(epochs):
            metric.reset_states()
            for c in callbacks:
                c.params['steps'] = train_steps_per_epoch
                c.on_epoch_begin(epoch)
            for idx, ((words, feats), (arcs, rels)) in enumerate(iter(trn_data)):
                logs = {}
                for c in callbacks:
                    c.on_batch_begin(idx, logs)
                mask = tf.not_equal(words, self.config.pad_index) & tf.not_equal(words, self.config.bos_index)
                loss, arc_scores, rel_scores = self.train_batch(words, feats, arcs, rels, mask,
                                                                optimizer, arc_loss, rel_loss)
                self.run_metrics(arcs, rels, arc_scores, rel_scores, words, mask, metric)
                logs['loss'] = loss
                logs.update(metric.to_dict())
                if epoch == epochs - 1:
                    self.model.stop_training = True
                for c in callbacks:
                    c.on_batch_end(idx, logs)
            # evaluate on dev
            metric.reset_states()
            logs = {}
            for idx, ((words, feats), (arcs, rels)) in enumerate(iter(dev_data)):
                arc_scores, rel_scores, loss, mask, arc_preds, rel_preds = self.evaluate_batch(words, feats, arcs, rels,
                                                                                               arc_loss, rel_loss,
                                                                                               metric)
                logs['val_loss'] = loss
                logs.update((f'val_{k}', v) for k, v in metric.to_dict().items())

            for c in callbacks:
                c.on_epoch_end(epoch, logs)
            if getattr(self.model, 'stop_training', None):
                break

        for c in callbacks:
            c.on_train_end()

    def evaluate(self, input_path: str, save_dir=None, output=False, batch_size=None, logger: logging.Logger = None,
                 callbacks: List[tf.keras.callbacks.Callback] = None, warm_up=False, verbose=True, **kwargs):
        if batch_size is None:
            batch_size = self.config.batch_size
        return super().evaluate(input_path, save_dir, output, batch_size, logger, callbacks, warm_up, verbose, **kwargs)

    def evaluate_batch(self, words, feats, arcs, rels, arc_loss, rel_loss, metric):
        mask = tf.not_equal(words, self.config.pad_index) & tf.not_equal(words, self.config.bos_index)
        arc_scores, rel_scores = self.model((words, feats))
        loss = self.get_loss(arc_scores, rel_scores, arcs, rels, mask, arc_loss, rel_loss)
        arc_preds, rel_preds = self.run_metrics(arcs, rels, arc_scores, rel_scores, words, mask, metric)
        return arc_scores, rel_scores, loss, mask, arc_preds, rel_preds

    def _build_metrics(self):
        if isinstance(self.config.metrics, tuple):
            self.config.metrics = list(self.config.metrics)
        if self.config.metrics == ['UAS', 'LAS']:
            metric = LabeledScore()
        else:
            metric = LabeledF1TF()
        return metric

    def run_metrics(self, arcs, rels, arc_scores, rel_scores, words, mask, metric):
        arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask)
        # ignore all punctuation if not specified
        if not self.config.punct:
            mask &= tf.reduce_all(tf.not_equal(tf.expand_dims(words, axis=-1), self.transform.puncts), axis=-1)
        metric(arc_preds, rel_preds, arcs, rels, mask)
        return arc_preds, rel_preds

    def train_batch(self, words, feats, arcs, rels, mask, optimizer, arc_loss, rel_loss):
        with tf.GradientTape() as tape:
            arc_scores, rel_scores = self.model((words, feats), training=True)
            loss = self.get_loss(arc_scores, rel_scores, arcs, rels, mask, arc_loss, rel_loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, arc_scores, rel_scores

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask, arc_loss, rel_loss):
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores = tf.gather_nd(rel_scores, tf.stack([tf.range(len(arcs), dtype=tf.int64), arcs], axis=1))
        arc_loss = arc_loss(arcs, arc_scores)
        rel_loss = rel_loss(rels, rel_scores)
        loss = arc_loss + rel_loss

        return loss

    def build_optimizer(self, optimizer='adam', lr=2e-3, mu=.9, nu=.9, epsilon=1e-12, clip=5.0, decay=.75,
                        decay_steps=5000, **kwargs):
        if optimizer == 'adam':
            scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                       decay_steps=decay_steps,
                                                                       decay_rate=decay)
            optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler,
                                                 beta_1=mu,
                                                 beta_2=nu,
                                                 epsilon=epsilon,
                                                 clipnorm=clip)
            return optimizer
        return super().build_optimizer(optimizer, **kwargs)

    # noinspection PyMethodOverriding
    def build_loss(self, arc_loss, rel_loss, **kwargs):
        if arc_loss == 'binary_crossentropy':
            arc_loss = tf.losses.BinaryCrossentropy(from_logits=True)
        else:
            arc_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True) if arc_loss == 'sparse_categorical_crossentropy' else super().build_loss(arc_loss)
        rel_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True) if rel_loss == 'sparse_categorical_crossentropy' else super().build_loss(rel_loss)
        return arc_loss, rel_loss

    @property
    def sample_data(self):
        return tf.constant([[2, 3, 4], [2, 5, 0]], dtype=tf.int64), tf.constant([[1, 2, 3], [4, 5, 0]], dtype=tf.int64)

    def num_samples_in(self, dataset):
        return sum(len(x[0][0]) for x in iter(dataset))

    def build_train_dataset(self, trn_data, batch_size, num_examples):
        trn_data = self.transform.file_to_dataset(trn_data, batch_size=batch_size,
                                                  shuffle=True,
                                                  repeat=None)
        return trn_data

    # noinspection PyMethodOverriding
    def build_callbacks(self, save_dir, logger, metrics, **kwargs):
        callbacks = super().build_callbacks(save_dir, logger, metrics=metrics, **kwargs)
        if isinstance(metrics, tuple):
            metrics = list(metrics)
        callbacks.append(self.build_progbar(metrics))
        params = {'verbose': 1, 'epochs': 1}
        for c in callbacks:
            c.set_params(params)
            c.set_model(self.model)
        return callbacks

    def build_progbar(self, metrics, training=True):
        return tf.keras.callbacks.ProgbarLogger(count_mode='steps',
                                                stateful_metrics=metrics + [f'val_{k}' for k in metrics] if training
                                                else [])

    def decode(self, arc_scores, rel_scores, mask):
        if self.config.tree:
            root_rel_idx = self.transform.root_rel_idx
            root_rel_onehot = np.eye(len(self.transform.rel_vocab))[root_rel_idx]
            arc_preds = np.zeros_like(mask, dtype=np.int64)
            rel_preds = np.zeros_like(mask, dtype=np.int64)
            for arc, rel, m, arc_pred, rel_pred in zip(arc_scores, rel_scores, mask, arc_preds, rel_preds):
                length = int(tf.math.count_nonzero(m)) + 1
                arc = arc[:length, :length]
                arc_probs = tf.nn.softmax(arc).numpy()
                m = np.expand_dims(m.numpy()[:length], -1)
                if self.config.tree == 'tarjan':
                    heads = tarjan(arc_probs, length, m)
                elif self.config.tree == 'mst':
                    heads, head_probs, tokens = unique_root(arc_probs, m, length)
                    arc = arc.numpy()
                    adjust_root_score(arc, heads, root_rel_idx)
                    heads = chu_liu_edmonds(arc, length)
                else:
                    raise ValueError(f'Unknown tree algorithm {self.config.tree}')
                arc_pred[:length] = heads
                root = np.where(heads[np.arange(1, length)] == 0)[0] + 1
                rel_prob = tf.nn.softmax(rel[:length, :length, :]).numpy()
                rel_prob = rel_prob[np.arange(length), heads]
                rel_prob[root] = root_rel_onehot
                rel_prob[np.arange(length) != root, np.arange(len(self.transform.rel_vocab)) == root_rel_idx] = 0
                # rels = rel_argmax(rel_prob, length, root_rel_idx)
                rels = np.argmax(rel_prob, axis=1)
                rel_pred[:length] = rels
            arc_preds = tf.constant(arc_preds)
            rel_preds = tf.constant(rel_preds)
        else:
            arc_preds = tf.argmax(arc_scores, -1)
            rel_preds = tf.argmax(rel_scores, -1)
            rel_preds = tf.squeeze(tf.gather(rel_preds, tf.expand_dims(arc_preds, -1), batch_dims=2), axis=-1)

        return arc_preds, rel_preds

    def evaluate_dataset(self, tst_data, callbacks, output, num_batches, ret_scores=None, **kwargs):
        if 'mask_p' in self.config:
            self.config['mask_p'] = None
        arc_loss, rel_loss = self.build_loss(**self.config)
        callbacks = [self.build_progbar(self.config['metrics'])]
        steps_per_epoch = len(list(iter(tst_data)))
        metric = self._build_metrics()
        params = {'verbose': 1, 'epochs': 1, 'metrics': ['loss'] + self.config.metrics, 'steps': steps_per_epoch}
        for c in callbacks:
            c.set_params(params)
            c.on_test_begin()
            c.on_epoch_end(0)
        logs = {}
        if ret_scores:
            scores = []
        if output:
            ext = os.path.splitext(output)[-1]
            output = open(output, 'w', encoding='utf-8')
        for idx, ((words, feats), Y) in enumerate(iter(tst_data)):
            arcs, rels = Y[0], Y[1]
            for c in callbacks:
                c.on_test_batch_begin(idx, logs)
            arc_scores, rel_scores, loss, mask, arc_preds, rel_preds = self.evaluate_batch(words, feats, arcs, rels,
                                                                                           arc_loss, rel_loss, metric)
            if ret_scores:
                scores.append((arc_scores.numpy(), rel_scores.numpy(), mask.numpy()))
            if output:
                for sent in self.transform.XY_to_inputs_outputs((words, feats, mask), (arc_preds, rel_preds),
                                                                conll=ext, arc_scores=arc_scores,
                                                                rel_scores=rel_scores):
                    output.write(str(sent))
                    output.write('\n\n')
            logs['loss'] = loss
            logs.update(metric.to_dict())
            for c in callbacks:
                c.on_test_batch_end(idx, logs)
        for c in callbacks:
            c.on_epoch_end(0)
            c.on_test_end()
        if output:
            output.close()
        loss = float(c.progbar._values['loss'][0] / c.progbar._values['loss'][1])
        outputs = loss, metric.to_dict(), False
        if ret_scores:
            outputs += (scores,)
        return outputs

    def predict_batch(self, batch, inputs=None, conll=True, **kwargs):
        ((words, feats), (arcs, rels)) = batch
        mask = tf.not_equal(words, self.config.pad_index) & tf.not_equal(words, self.config.bos_index)
        arc_scores, rel_scores = self.model((words, feats))
        arc_preds, rel_preds = self.decode(arc_scores, rel_scores, mask)
        for sent in self.transform.XY_to_inputs_outputs((words, feats, mask), (arc_preds, rel_preds), gold=False,
                                                        inputs=inputs, conll=conll):
            yield sent

    def compile_model(self, optimizer, loss, metrics):
        super().compile_model(optimizer, loss, metrics)


class BiaffineSemanticDependencyParserTF(BiaffineDependencyParserTF):
    def __init__(self, transform: CoNLL_SDP_Transform = None) -> None:
        if not transform:
            transform = CoNLL_SDP_Transform()
        # noinspection PyTypeChecker
        super().__init__(transform)
        self.transform: CoNLL_SDP_Transform = transform

    def fit(self, trn_data, dev_data, save_dir, n_embed=100, pretrained_embed=None, embed_dropout=.33,
            n_lstm_hidden=400, n_lstm_layers=3, lstm_dropout=.33, n_mlp_arc=500, n_mlp_rel=100, mlp_dropout=.33,
            optimizer='adam', lr=2e-3, mu=.9, nu=.9, epsilon=1e-12, clip=5.0, decay=.75, decay_steps=5000, patience=100,
            arc_loss='binary_crossentropy', rel_loss='sparse_categorical_crossentropy',
            metrics=('UF', 'LF'), n_buckets=32, batch_size=5000, epochs=50000, early_stopping_patience=100,
            tree=False, punct=False, min_freq=2, run_eagerly=False, logger=None, verbose=True, **kwargs):
        return super().fit(trn_data, dev_data, save_dir, n_embed, pretrained_embed, embed_dropout, n_lstm_hidden,
                           n_lstm_layers, lstm_dropout, n_mlp_arc, n_mlp_rel, mlp_dropout, optimizer, lr, mu, nu,
                           epsilon, clip, decay, decay_steps, patience, arc_loss, rel_loss, metrics, n_buckets,
                           batch_size, epochs, early_stopping_patience, tree, punct, min_freq, run_eagerly, logger,
                           verbose, **kwargs)

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask, arc_loss, rel_loss):
        mask = tf.tile(tf.expand_dims(mask, -1), [1, 1, tf.shape(mask)[-1]])
        mask &= tf.transpose(mask, [0, 2, 1])
        arc_scores, arcs = arc_scores[mask], arcs[mask]
        rel_scores, rels = rel_scores[mask], rels[mask]
        rel_scores, rels = rel_scores[arcs], rels[arcs]
        arc_loss = arc_loss(arcs, arc_scores)
        rel_loss = rel_loss(rels, rel_scores)
        loss = arc_loss + rel_loss

        return loss

    def decode(self, arc_scores, rel_scores, mask):
        arc_preds = arc_scores > 0
        rel_preds = tf.argmax(rel_scores, -1)

        return arc_preds, rel_preds


class BiaffineTransformerDependencyParserTF(BiaffineDependencyParserTF, tf.keras.callbacks.Callback):
    def __init__(self, transform: CoNLL_Transformer_Transform = None) -> None:
        if not transform:
            transform = CoNLL_Transformer_Transform()
        super().__init__(transform)
        self.transform: CoNLL_Transformer_Transform = transform

    def build_model(self, transformer, training, **kwargs) -> tf.keras.Model:
        transformer = self.build_transformer(training, transformer)
        model = BiaffineModelTF(self.config, transformer=transformer)
        return model

    def build_transformer(self, training, transformer):
        if training:
            self.config.n_words = len(self.transform.form_vocab)
        self._init_config()
        if isinstance(transformer, str):
            if 'albert_chinese' in transformer:
                tokenizer = BertTokenizerFast.from_pretrained(transformer, add_special_tokens=False)
                transformer: TFPreTrainedModel = TFAutoModel.from_pretrained(transformer, name=transformer,
                                                                             from_pt=True)
            elif transformer.startswith('albert') and transformer.endswith('zh'):
                transformer, tokenizer, path = build_transformer(transformer)
                transformer.config = AlbertConfig.from_json_file(os.path.join(path, "albert_config.json"))
                tokenizer = BertTokenizer.from_pretrained(os.path.join(path, "vocab_chinese.txt"),
                                                          add_special_tokens=False)
            elif 'chinese-roberta' in transformer:
                tokenizer = BertTokenizer.from_pretrained(transformer)
                transformer = TFBertModel.from_pretrained(transformer, name=transformer, from_pt=True)
            else:
                tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(transformer)
                try:
                    transformer: TFPreTrainedModel = TFAutoModel.from_pretrained(transformer, name=transformer)
                except (TypeError, OSError):
                    transformer: TFPreTrainedModel = TFAutoModel.from_pretrained(transformer, name=transformer,
                                                                                 from_pt=True)
        elif transformer[0] == 'AutoModelWithLMHead':
            tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(transformer[1])
            transformer: TFAutoModelWithLMHead = TFAutoModelWithLMHead.from_pretrained(transformer[1])
        else:
            raise ValueError(f'Unknown identifier {transformer}')
        self.transform.tokenizer = tokenizer
        if self.config.get('fp16', None) or self.config.get('use_amp', None):
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
            tf.keras.mixed_precision.experimental.set_policy(policy)
            # tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            transformer.set_weights([w.astype('float16') for w in transformer.get_weights()])
        self.transform.transformer_config = transformer.config
        return transformer

    # noinspection PyMethodOverriding
    def fit(self, trn_data, dev_data, save_dir, transformer, max_seq_length=256, transformer_dropout=.33,
            d_positional=None,
            n_mlp_arc=500, n_mlp_rel=100, mlp_dropout=.33,
            optimizer='adamw',
            learning_rate=5e-5,
            learning_rate_transformer=None,
            weight_decay_rate=0,
            epsilon=1e-8,
            clipnorm=None,
            fp16=False,
            warmup_steps_ratio=0,
            arc_loss='sparse_categorical_crossentropy', rel_loss='sparse_categorical_crossentropy',
            metrics=('UAS', 'LAS'),
            batch_size=3000,
            samples_per_batch=150,
            max_samples_per_batch=None,
            epochs=100,
            tree=False, punct=False, token_mapping=None, run_eagerly=False, logger=None, verbose=True, **kwargs):
        self.set_params({})
        return KerasComponent.fit(self, **merge_locals_kwargs(locals(), kwargs))

    @property
    def sample_data(self):
        dataset = self.transform.inputs_to_dataset(
            [[('Hello', 'NN'), ('world', 'NN')], [('HanLP', 'NN'), ('is', 'NN'), ('good', 'NN')]] if self.config.get(
                'use_pos', None) else
            [['Hello', 'world'], ['HanLP', 'is', 'good']])
        return next(iter(dataset))[0]

    # noinspection PyMethodOverriding
    def build_optimizer(self, optimizer, learning_rate, epsilon, weight_decay_rate, clipnorm, fp16, train_steps,
                        **kwargs):
        if optimizer == 'adamw':
            epochs = self.config['epochs']
            learning_rate_transformer = kwargs.get('learning_rate_transformer', None)
            train_steps = math.ceil(self.config.train_examples * epochs / self.config.samples_per_batch)
            warmup_steps = math.ceil(train_steps * self.config['warmup_steps_ratio'])
            if learning_rate_transformer is not None:
                if learning_rate_transformer > 0:
                    self.params['optimizer_transformer'] = build_adamw_optimizer(self.config, learning_rate_transformer,
                                                                                 epsilon,
                                                                                 clipnorm, train_steps, fp16,
                                                                                 math.ceil(warmup_steps),
                                                                                 weight_decay_rate)
                else:
                    self.model.transformer.trainable = False
                return super().build_optimizer(lr=learning_rate)  # use a normal adam for biaffine
            else:
                return build_adamw_optimizer(self.config, learning_rate, epsilon, clipnorm, train_steps, fp16,
                                             math.ceil(warmup_steps), weight_decay_rate)
        return super().build_optimizer(optimizer, **kwargs)

    def build_vocab(self, trn_data, logger):
        self.config.train_examples = train_examples = super().build_vocab(trn_data, logger)
        return train_examples

    def build_callbacks(self, save_dir, logger, metrics, **kwargs):
        callbacks = super().build_callbacks(save_dir, logger, metrics=metrics, **kwargs)
        callbacks.append(self)
        if not self.params:
            self.set_params({})
        return callbacks

    def on_train_begin(self):
        self.params['accum_grads'] = [tf.Variable(tf.zeros_like(tv.read_value()), trainable=False) for tv in
                                      self.model.trainable_variables]
        self.params['trained_samples'] = 0
        self.params['transformer_variable_names'] = {x.name for x in self.model.transformer.trainable_variables}

    def train_batch(self, words, feats, arcs, rels, mask, optimizer, arc_loss, rel_loss):
        with tf.GradientTape() as tape:
            arc_scores, rel_scores = self.model((words, feats), training=True)
            loss = self.get_loss(arc_scores, rel_scores, arcs, rels, mask, arc_loss, rel_loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        accum_grads = self.params['accum_grads']
        for i, grad in enumerate(grads):
            if grad is not None:
                accum_grads[i].assign_add(grad)
        self.params['trained_samples'] += tf.shape(words)[0]
        if self.params['trained_samples'] >= self.config.samples_per_batch:
            self._apply_grads(accum_grads)
        return loss, arc_scores, rel_scores

    def _apply_grads(self, accum_grads):
        optimizer_transformer = self.params.get('optimizer_transformer', None)
        if optimizer_transformer:
            transformer = self.params['transformer_variable_names']
            trainable_variables = self.model.trainable_variables
            optimizer_transformer.apply_gradients(
                (g, w) for g, w in zip(accum_grads, trainable_variables) if w.name in transformer)
            self.model.optimizer.apply_gradients(
                (g, w) for g, w in zip(accum_grads, trainable_variables) if w.name not in transformer)
        else:
            self.model.optimizer.apply_gradients(zip(accum_grads, self.model.trainable_variables))
        for tv in accum_grads:
            tv.assign(tf.zeros_like(tv))
        # print('Apply grads after', self.params['trained_samples'], 'samples')
        self.params['trained_samples'] = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.params['trained_samples']:
            self._apply_grads(self.params['accum_grads'])


class BiaffineTransformerSemanticDependencyParser(BiaffineTransformerDependencyParserTF):

    def __init__(self, transform: CoNLL_Transformer_Transform = None) -> None:
        if not transform:
            transform = CoNLL_Transformer_Transform(graph=True)
        super().__init__(transform)

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask, arc_loss, rel_loss):
        return BiaffineSemanticDependencyParserTF.get_loss(self, arc_scores, rel_scores, arcs, rels, mask, arc_loss,
                                                           rel_loss)

    def fit(self, trn_data, dev_data, save_dir, transformer, max_seq_length=256, transformer_dropout=.33,
            d_positional=None, n_mlp_arc=500, n_mlp_rel=100, mlp_dropout=.33, optimizer='adamw', learning_rate=5e-5,
            learning_rate_transformer=None, weight_decay_rate=0, epsilon=1e-8, clipnorm=None, fp16=False,
            warmup_steps_ratio=0, arc_loss='binary_crossentropy',
            rel_loss='sparse_categorical_crossentropy', metrics=('UF', 'LF'), batch_size=3000, samples_per_batch=150,
            max_samples_per_batch=None, epochs=100, tree=False, punct=False, token_mapping=None, enhanced_only=False,
            run_eagerly=False,
            logger=None, verbose=True, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def decode(self, arc_scores, rel_scores, mask):
        return BiaffineSemanticDependencyParserTF.decode(self, arc_scores, rel_scores, mask)


class StructuralAttentionDependencyParserTF(BiaffineTransformerDependencyParserTF):

    def build_model(self, transformer, training, masked_lm_embed=None, **kwargs) -> tf.keras.Model:
        transformer = self.build_transformer(training, transformer)
        self.config.num_heads = len(self.transform.rel_vocab)
        if self.config.get('use_pos', None):
            self.config.n_pos = len(self.transform.cpos_vocab)
        if masked_lm_embed:
            masked_lm_embed = build_embedding(masked_lm_embed, self.transform.form_vocab, self.transform)
            masked_lm_embed(tf.constant(0))  # build it with sample data
            masked_lm_embed = tf.transpose(masked_lm_embed._embeddings)
        return StructuralAttentionModel(self.config, transformer, masked_lm_embed)

    def fit(self, trn_data, dev_data, save_dir, transformer, max_seq_length=256, transformer_dropout=.33,
            d_positional=None, mask_p=.15, masked_lm_dropout=None, masked_lm_embed=None, joint_pos=False, alpha=0.1,
            sa_dim=None,
            num_decoder_layers=1,
            n_mlp_arc=500,
            n_mlp_rel=100,
            mlp_dropout=.33,
            optimizer='adamw',
            learning_rate=5e-5,
            learning_rate_transformer=None, weight_decay_rate=0, epsilon=1e-8, clipnorm=None, fp16=False,
            warmup_steps_ratio=0, arc_loss='sparse_categorical_crossentropy',
            rel_loss='sparse_categorical_crossentropy', metrics=('UAS', 'LAS'), batch_size=3000, samples_per_batch=150,
            epochs=100, tree=False, punct=False, token_mapping=None, run_eagerly=False, logger=None, verbose=True,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def train_loop(self, trn_data, dev_data, epochs, num_examples, train_steps_per_epoch, dev_steps, model, optimizer,
                   loss, metrics, callbacks, logger: logging.Logger, arc_loss, rel_loss, **kwargs):
        arc_loss, rel_loss = loss
        # because we are customizing batching
        train_steps_per_epoch = len(list(iter(trn_data)))
        # progbar: tf.keras.callbacks.ProgbarLogger = callbacks[-1]
        c: tf.keras.callbacks.Callback = None
        metrics = self._build_metrics()
        acc: tf.keras.metrics.SparseCategoricalAccuracy = metrics[1]
        for c in callbacks:
            if not hasattr(c, 'params'):
                c.params = {}
            c.params['epochs'] = epochs
            c.params['trn_data'] = trn_data
            c.params['metrics'] = ['loss'] + self.config.metrics + [acc.name]
            c.params['metrics'] = c.params['metrics'] + [f'val_{k}' for k in c.params['metrics']]
            c.on_train_begin()
        for epoch in range(epochs):
            for metric in metrics:
                metric.reset_states()
            for c in callbacks:
                c.params['steps'] = train_steps_per_epoch
                c.on_epoch_begin(epoch)
            for idx, ((words, feats), (arcs, rels, offsets)) in enumerate(iter(trn_data)):
                logs = {}
                for c in callbacks:
                    c.on_batch_begin(idx, logs)
                mask = tf.not_equal(words, self.config.pad_index) & tf.not_equal(words, self.config.bos_index)
                loss, arc_scores, rel_scores, lm_ids = self.train_batch(words, feats, arcs, rels, offsets, mask,
                                                                        optimizer, arc_loss, rel_loss, acc)
                self.run_metrics(arcs, rels, arc_scores, rel_scores, words, mask, metrics[0])
                logs['loss'] = loss
                logs.update(metrics[0].to_dict())
                logs[acc.name] = acc.result()
                if epoch == epochs - 1:
                    self.model.stop_training = True
                for c in callbacks:
                    c.on_batch_end(idx, logs)
            # evaluate on dev
            for metric in metrics:
                metric.reset_states()
            logs = {}
            for idx, ((words, feats), (arcs, rels, offsets)) in enumerate(iter(dev_data)):
                arc_scores, rel_scores, loss, mask, arc_preds, rel_preds = self.evaluate_batch(words, feats, arcs, rels,
                                                                                               arc_loss, rel_loss,
                                                                                               metrics[0])
                logs['val_loss'] = loss
                logs.update((f'val_{k}', v) for k, v in metrics[0].to_dict().items())

            for c in callbacks:
                c.on_epoch_end(epoch, logs)
            if getattr(self.model, 'stop_training', None):
                break

        for c in callbacks:
            c.on_train_end()

    # noinspection PyMethodOverriding
    def train_batch(self, words, feats, arcs, rels, ids, mask, optimizer, arc_loss, rel_loss, metric):
        with tf.GradientTape() as tape:
            arc_scores, rel_scores, lm_ids = self.model((words, feats), training=True)
            loss = self.get_total_loss(words, feats, arcs, rels, arc_scores, rel_scores, arc_loss, rel_loss, ids,
                                       lm_ids, mask,
                                       metric)
        grads = tape.gradient(loss, self.model.trainable_variables)
        accum_grads = self.params['accum_grads']
        for i, grad in enumerate(grads):
            if grad is not None:
                accum_grads[i].assign_add(grad)
        self.params['trained_samples'] += tf.shape(words)[0]
        if self.params['trained_samples'] >= self.config.samples_per_batch:
            self._apply_grads(accum_grads)
        return loss, arc_scores, rel_scores, lm_ids

    def get_total_loss(self, words, feats, arcs, rels, arc_scores, rel_scores, arc_loss, rel_loss, gold_offsets,
                       pred_ids,
                       mask, metric):
        masked_lm_loss = self.get_masked_lm_loss(words, feats, gold_offsets, pred_ids, metric)
        # return masked_lm_loss
        parser_loss = self.get_loss(arc_scores, rel_scores, arcs, rels, mask, arc_loss, rel_loss)
        loss = parser_loss + masked_lm_loss * self.config.alpha
        return loss

    def get_masked_lm_loss(self, words, feats, gold_offsets, pred_ids, metric):
        if self.config.get('joint_pos', None):
            gold_ids = tf.gather(feats[-1], gold_offsets, batch_dims=1)
        else:
            gold_ids = tf.gather(words, gold_offsets, batch_dims=1)
        pred_ids = tf.gather(pred_ids, gold_offsets, batch_dims=1)
        masked_lm_loss = tf.keras.losses.sparse_categorical_crossentropy(gold_ids, pred_ids)
        mask = gold_offsets != 0
        if metric:
            metric(gold_ids, pred_ids, mask)
        return tf.reduce_mean(tf.boolean_mask(masked_lm_loss, mask))

    def _build_metrics(self):
        if not self.config['mask_p']:
            return super()._build_metrics()
        acc = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        return super()._build_metrics(), acc

    def build_train_dataset(self, trn_data, batch_size, num_examples):
        trn_data = self.transform.file_to_dataset(trn_data, batch_size=batch_size,
                                                  shuffle=True,
                                                  repeat=None,
                                                  cache=False)  # Generate different masks every time
        return trn_data

    def build_loss(self, arc_loss, rel_loss, **kwargs):
        if arc_loss == 'binary_crossentropy':
            arc_loss = tf.losses.BinaryCrossentropy(from_logits=False)
        else:
            arc_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True) if arc_loss == 'sparse_categorical_crossentropy' else super().build_loss(arc_loss)
        rel_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True) if rel_loss == 'sparse_categorical_crossentropy' else super().build_loss(rel_loss)
        return arc_loss, rel_loss

    def decode(self, arc_scores, rel_scores, mask):
        if self.transform.graph:
            return BiaffineSemanticDependencyParserTF.decode(self, arc_scores, rel_scores, mask)
        return super().decode(arc_scores, rel_scores, mask)

    def get_loss(self, arc_scores, rel_scores, arcs, rels, mask, arc_loss, rel_loss):
        if self.transform.graph:
            return BiaffineSemanticDependencyParserTF.get_loss(self, arc_scores, rel_scores, arcs, rels, mask, arc_loss,
                                                               rel_loss)
        return super().get_loss(arc_scores, rel_scores, arcs, rels, mask, arc_loss, rel_loss)
