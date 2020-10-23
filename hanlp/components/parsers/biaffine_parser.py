# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-22 12:47
import logging
import tensorflow as tf
from typing import List, Tuple, Union
from hanlp.common.component import KerasComponent
from hanlp.components.parsers.biaffine.model import BiaffineModel
from hanlp.components.parsers.conll import CoNLLSentence, CoNLL_DEP_Transform, CoNLL_SDP_Transform
from hanlp.layers.embeddings import build_embedding
from hanlp.metrics.parsing.labeled_f1 import LabeledF1
from hanlp.metrics.parsing.labeled_score import LabeledScore
from hanlp.utils.util import merge_locals_kwargs


class BiaffineDependencyParser(KerasComponent):
    def __init__(self, transform: CoNLL_DEP_Transform = None) -> None:
        if not transform:
            transform = CoNLL_DEP_Transform()
        super().__init__(transform)
        self.transform: CoNLL_DEP_Transform = transform
        self.model: BiaffineModel = None

    def build_model(self, pretrained_embed, n_embed, training, **kwargs) -> tf.keras.Model:
        if training:
            self.config.n_words = len(self.transform.form_vocab)
        else:
            self.config.lstm_dropout = 0.  # keras will use cuda lstm when config.lstm_dropout is 0
        self.config.n_feats = len(self.transform.cpos_vocab)
        self.config.n_rels = len(self.transform.rel_vocab)
        self.config.pad_index = self.transform.form_vocab.pad_idx
        self.config.unk_index = self.transform.form_vocab.unk_idx
        self.config.bos_index = 2
        pretrained: tf.keras.layers.Embedding = build_embedding(pretrained_embed, self.transform.form_vocab,
                                                                self.transform) if pretrained_embed else None
        if pretrained_embed:
            self.config.n_embed = pretrained.output_dim
        model = BiaffineModel(self.config, pretrained)
        return model

    def load_weights(self, save_dir, filename='model.h5', functional=False, **kwargs):
        super().load_weights(save_dir, filename)
        if functional:
            self.model = self.model.to_functional()

    def build_vocab(self, trn_data, logger):
        return super().build_vocab(trn_data, logger)

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
                params = {'verbose': 1, 'epochs': epochs, 'steps': train_steps_per_epoch}
                c.params = params
                c.set_params(params)
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
                    c.on_train_batch_end(idx, logs)
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
                 callbacks: List[tf.keras.callbacks.Callback] = None, warm_up=True, verbose=True, **kwargs):
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
            metric = LabeledF1()
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

    def build_optimizer(self, optimizer, **kwargs):
        if optimizer == 'adam':
            scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.config.lr,
                                                                       decay_steps=self.config.decay_steps,
                                                                       decay_rate=self.config.decay)
            optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler,
                                                 beta_1=self.config.mu,
                                                 beta_2=self.config.nu,
                                                 epsilon=self.config.epsilon,
                                                 clipnorm=self.config.clip)
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
        params = {'verbose': 1, 'epochs': self.config.epochs}
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
            # arc_preds = eisner(arc_scores, mask)
            pass
        else:
            arc_preds = tf.argmax(arc_scores, -1)

        rel_preds = tf.argmax(rel_scores, -1)
        rel_preds = tf.squeeze(tf.gather(rel_preds, tf.expand_dims(arc_preds, -1), batch_dims=2), axis=-1)

        return arc_preds, rel_preds

    def evaluate_dataset(self, tst_data, callbacks, output, num_batches):
        arc_loss, rel_loss = self.build_loss(**self.config)
        callbacks = [self.build_progbar(self.config['metrics'])]
        steps_per_epoch = len(list(iter(tst_data)))
        metric = self._build_metrics()
        params = {'verbose': 1, 'epochs': 1, 'metrics': ['loss'] + self.config.metrics, 'steps': steps_per_epoch}
        for c in callbacks:
            c.set_params(params)
            c.on_train_begin()  # otherwise AttributeError: 'ProgbarLogger' object has no attribute 'verbose'
            c.on_epoch_begin(0)
        logs = {}
        if output:
            output = open(output, 'w', encoding='utf-8')
        for idx, ((words, feats), (arcs, rels)) in enumerate(iter(tst_data)):
            for c in callbacks:
                c.on_batch_begin(idx, logs)
            arc_scores, rel_scores, loss, mask, arc_preds, rel_preds = self.evaluate_batch(words, feats, arcs, rels,
                                                                                           arc_loss, rel_loss, metric)
            if output:
                for sent in self.transform.XY_to_inputs_outputs((words, feats, mask), (arc_preds, rel_preds)):
                    output.write(str(sent))
                    output.write('\n\n')
            logs['loss'] = loss
            logs.update(metric.to_dict())
            for c in callbacks:
                c.on_batch_end(idx, logs)
        for c in callbacks:
            c.on_epoch_end(0)
            c.on_test_end()
        if output:
            output.close()
        loss = float(c.progbar._values['loss'][0] / c.progbar._values['loss'][1])
        return loss, metric.to_dict(), False

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


class BiaffineSemanticDependencyParser(BiaffineDependencyParser):
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
        if self.config.tree:
            # arc_preds = eisner(arc_scores, mask)
            raise NotImplemented('Give me some time...')
        else:
            arc_preds = arc_scores > 0

        rel_preds = tf.argmax(rel_scores, -1)

        return arc_preds, rel_preds
