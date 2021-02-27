# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-26 14:45
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict

import numpy as np
import tensorflow as tf

import hanlp.utils
from hanlp_common.io import save_json,load_json
from hanlp.callbacks.fine_csv_logger import FineCSVLogger
from hanlp.common.component import Component
from hanlp.common.transform_tf import Transform
from hanlp.common.vocab_tf import VocabTF
from hanlp.metrics.chunking.iobes_tf import IOBES_F1_TF
from hanlp.optimizers.adamw import AdamWeightDecay
from hanlp.utils import io_util
from hanlp.utils.io_util import get_resource, tempdir_human
from hanlp.utils.log_util import init_logger, logger
from hanlp.utils.string_util import format_scores
from hanlp.utils.tf_util import format_metrics, size_of_dataset, summary_of_model, get_callback_by_class
from hanlp.utils.time_util import Timer, now_datetime
from hanlp_common.reflection import str_to_type, classpath_of
from hanlp_common.structure import SerializableDict
from hanlp_common.util import merge_dict


class KerasComponent(Component, ABC):
    def __init__(self, transform: Transform) -> None:
        super().__init__()
        self.meta = {
            'class_path': classpath_of(self),
            'hanlp_version': hanlp.version.__version__,
        }
        self.model: Optional[tf.keras.Model] = None
        self.config = SerializableDict()
        self.transform = transform
        # share config with transform for convenience, so we don't need to pass args around
        if self.transform.config:
            for k, v in self.transform.config.items():
                self.config[k] = v
        self.transform.config = self.config

    def evaluate(self, input_path: str, save_dir=None, output=False, batch_size=128, logger: logging.Logger = None,
                 callbacks: List[tf.keras.callbacks.Callback] = None, warm_up=True, verbose=True, **kwargs):
        input_path = get_resource(input_path)
        file_prefix, ext = os.path.splitext(input_path)
        name = os.path.basename(file_prefix)
        if not name:
            name = 'evaluate'
        if save_dir and not logger:
            logger = init_logger(name=name, root_dir=save_dir, level=logging.INFO if verbose else logging.WARN,
                                 mode='w')
        tst_data = self.transform.file_to_dataset(input_path, batch_size=batch_size)
        samples = self.num_samples_in(tst_data)
        num_batches = math.ceil(samples / batch_size)
        if warm_up:
            for x, y in tst_data:
                self.model.predict_on_batch(x)
                break
        if output:
            assert save_dir, 'Must pass save_dir in order to output'
            if isinstance(output, bool):
                output = os.path.join(save_dir, name) + '.predict' + ext
            elif isinstance(output, str):
                output = output
            else:
                raise RuntimeError('output ({}) must be of type bool or str'.format(repr(output)))
        timer = Timer()
        eval_outputs = self.evaluate_dataset(tst_data, callbacks, output, num_batches, **kwargs)
        loss, score, output = eval_outputs[0], eval_outputs[1], eval_outputs[2]
        delta_time = timer.stop()
        speed = samples / delta_time.delta_seconds

        if logger:
            f1: IOBES_F1_TF = None
            for metric in self.model.metrics:
                if isinstance(metric, IOBES_F1_TF):
                    f1 = metric
                    break
            extra_report = ''
            if f1:
                overall, by_type, extra_report = f1.state.result(full=True, verbose=False)
                extra_report = ' \n' + extra_report
            logger.info('Evaluation results for {} - '
                        'loss: {:.4f} - {} - speed: {:.2f} sample/sec{}'
                        .format(name + ext, loss,
                                format_scores(score) if isinstance(score, dict) else format_metrics(self.model.metrics),
                                speed, extra_report))
        if output:
            logger.info('Saving output to {}'.format(output))
            with open(output, 'w', encoding='utf-8') as out:
                self.evaluate_output(tst_data, out, num_batches, self.model.metrics)

        return loss, score, speed

    def num_samples_in(self, dataset):
        return size_of_dataset(dataset)

    def evaluate_dataset(self, tst_data, callbacks, output, num_batches, **kwargs):
        loss, score = self.model.evaluate(tst_data, callbacks=callbacks, steps=num_batches)
        return loss, score, output

    def evaluate_output(self, tst_data, out, num_batches, metrics: List[tf.keras.metrics.Metric]):
        # out.write('x\ty_true\ty_pred\n')
        for metric in metrics:
            metric.reset_states()
        for idx, batch in enumerate(tst_data):
            outputs = self.model.predict_on_batch(batch[0])
            for metric in metrics:
                metric(batch[1], outputs, outputs._keras_mask if hasattr(outputs, '_keras_mask') else None)
            self.evaluate_output_to_file(batch, outputs, out)
            print('\r{}/{} {}'.format(idx + 1, num_batches, format_metrics(metrics)), end='')
        print()

    def evaluate_output_to_file(self, batch, outputs, out):
        for x, y_gold, y_pred in zip(self.transform.X_to_inputs(batch[0]),
                                     self.transform.Y_to_outputs(batch[1], gold=True),
                                     self.transform.Y_to_outputs(outputs, gold=False)):
            out.write(self.transform.input_truth_output_to_str(x, y_gold, y_pred))

    def _capture_config(self, config: Dict,
                        exclude=(
                                'trn_data', 'dev_data', 'save_dir', 'kwargs', 'self', 'logger', 'verbose',
                                'dev_batch_size', '__class__')):
        """
        Save arguments to config

        Parameters
        ----------
        config
            `locals()`
        exclude
        """
        if 'kwargs' in config:
            config.update(config['kwargs'])
        config = dict(
            (key, tf.keras.utils.serialize_keras_object(value)) if hasattr(value, 'get_config') else (key, value) for
            key, value in config.items())
        for key in exclude:
            config.pop(key, None)
        self.config.update(config)

    def save_meta(self, save_dir, filename='meta.json', **kwargs):
        self.meta['create_time']: now_datetime()
        self.meta.update(kwargs)
        save_json(self.meta, os.path.join(save_dir, filename))

    def load_meta(self, save_dir, filename='meta.json'):
        save_dir = get_resource(save_dir)
        metapath = os.path.join(save_dir, filename)
        if os.path.isfile(metapath):
            self.meta.update(load_json(metapath))

    def save_config(self, save_dir, filename='config.json'):
        self.config.save_json(os.path.join(save_dir, filename))

    def load_config(self, save_dir, filename='config.json'):
        save_dir = get_resource(save_dir)
        self.config.load_json(os.path.join(save_dir, filename))

    def save_weights(self, save_dir, filename='model.h5'):
        self.model.save_weights(os.path.join(save_dir, filename))

    def load_weights(self, save_dir, filename='model.h5', **kwargs):
        assert self.model.built or self.model.weights, 'You must call self.model.built() in build_model() ' \
                                                       'in order to load it'
        save_dir = get_resource(save_dir)
        self.model.load_weights(os.path.join(save_dir, filename))

    def save_vocabs(self, save_dir, filename='vocabs.json'):
        vocabs = SerializableDict()
        for key, value in vars(self.transform).items():
            if isinstance(value, VocabTF):
                vocabs[key] = value.to_dict()
        vocabs.save_json(os.path.join(save_dir, filename))

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        save_dir = get_resource(save_dir)
        vocabs = SerializableDict()
        vocabs.load_json(os.path.join(save_dir, filename))
        for key, value in vocabs.items():
            vocab = VocabTF()
            vocab.copy_from(value)
            setattr(self.transform, key, vocab)

    def load_transform(self, save_dir) -> Transform:
        """
        Try to load transform only. This method might fail due to the fact it avoids building the model.
        If it do fail, then you have to use `load` which might be too heavy but that's the best we can do.
        :param save_dir: The path to load.
        """
        save_dir = get_resource(save_dir)
        self.load_config(save_dir)
        self.load_vocabs(save_dir)
        self.transform.build_config()
        self.transform.lock_vocabs()
        return self.transform

    def save(self, save_dir: str, **kwargs):
        self.save_config(save_dir)
        self.save_vocabs(save_dir)
        self.save_weights(save_dir)

    def load(self, save_dir: str, logger=hanlp.utils.log_util.logger, **kwargs):
        self.meta['load_path'] = save_dir
        save_dir = get_resource(save_dir)
        self.load_config(save_dir)
        self.load_vocabs(save_dir)
        self.build(**merge_dict(self.config, training=False, logger=logger, **kwargs, overwrite=True, inplace=True))
        self.load_weights(save_dir, **kwargs)
        self.load_meta(save_dir)

    @property
    def input_shape(self) -> List:
        return self.transform.output_shapes[0]

    def build(self, logger, **kwargs):
        self.transform.build_config()
        self.model = self.build_model(**merge_dict(self.config, training=kwargs.get('training', None),
                                                   loss=kwargs.get('loss', None)))
        self.transform.lock_vocabs()
        optimizer = self.build_optimizer(**self.config)
        loss = self.build_loss(
            **self.config if 'loss' in self.config else dict(list(self.config.items()) + [('loss', None)]))
        # allow for different
        metrics = self.build_metrics(**merge_dict(self.config, metrics=kwargs.get('metrics', 'accuracy'),
                                                  logger=logger, overwrite=True))
        if not isinstance(metrics, list):
            if isinstance(metrics, tf.keras.metrics.Metric):
                metrics = [metrics]
        if not self.model.built:
            sample_inputs = self.sample_data
            if sample_inputs is not None:
                self.model(sample_inputs)
            else:
                if len(self.transform.output_shapes[0]) == 1 and self.transform.output_shapes[0][0] is None:
                    x_shape = self.transform.output_shapes[0]
                else:
                    x_shape = list(self.transform.output_shapes[0])
                    for i, shape in enumerate(x_shape):
                        x_shape[i] = [None] + shape  # batch + X.shape
                self.model.build(input_shape=x_shape)
        self.compile_model(optimizer, loss, metrics)
        return self.model, optimizer, loss, metrics

    def compile_model(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=self.config.run_eagerly)

    def build_optimizer(self, optimizer, **kwargs):
        if isinstance(optimizer, (str, dict)):
            custom_objects = {'AdamWeightDecay': AdamWeightDecay}
            optimizer: tf.keras.optimizers.Optimizer = tf.keras.utils.deserialize_keras_object(optimizer,
                                                                                               module_objects=vars(tf.keras.optimizers),
                                                                                               custom_objects=custom_objects)
        self.config.optimizer = tf.keras.utils.serialize_keras_object(optimizer)
        return optimizer

    def build_loss(self, loss, **kwargs):
        if not loss:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                from_logits=True)
        elif isinstance(loss, (str, dict)):
            loss = tf.keras.utils.deserialize_keras_object(loss, module_objects=vars(tf.keras.losses))
        if isinstance(loss, tf.keras.losses.Loss):
            self.config.loss = tf.keras.utils.serialize_keras_object(loss)
        return loss

    def build_transform(self, **kwargs):
        return self.transform

    def build_vocab(self, trn_data, logger):
        train_examples = self.transform.fit(trn_data, **self.config)
        self.transform.summarize_vocabs(logger)
        return train_examples

    def build_metrics(self, metrics, logger: logging.Logger, **kwargs):
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        return [metric]

    @abstractmethod
    def build_model(self, **kwargs) -> tf.keras.Model:
        pass

    def fit(self, trn_data, dev_data, save_dir, batch_size, epochs, run_eagerly=False, logger=None, verbose=True,
            finetune: str = None, **kwargs):
        self._capture_config(locals())
        self.transform = self.build_transform(**self.config)
        if not save_dir:
            save_dir = tempdir_human()
        if not logger:
            logger = init_logger(name='train', root_dir=save_dir, level=logging.INFO if verbose else logging.WARN)
        logger.info('Hyperparameter:\n' + self.config.to_json())
        num_examples = self.build_vocab(trn_data, logger)
        # assert num_examples, 'You forgot to return the number of training examples in your build_vocab'
        logger.info('Building...')
        train_steps_per_epoch = math.ceil(num_examples / batch_size) if num_examples else None
        self.config.train_steps = train_steps_per_epoch * epochs if num_examples else None
        model, optimizer, loss, metrics = self.build(**merge_dict(self.config, logger=logger, training=True))
        logger.info('Model built:\n' + summary_of_model(self.model))
        if finetune:
            finetune = get_resource(finetune)
            if os.path.isdir(finetune):
                finetune = os.path.join(finetune, 'model.h5')
            model.load_weights(finetune, by_name=True, skip_mismatch=True)
            logger.info(f'Loaded pretrained weights from {finetune} for finetuning')
        self.save_config(save_dir)
        self.save_vocabs(save_dir)
        self.save_meta(save_dir)
        trn_data = self.build_train_dataset(trn_data, batch_size, num_examples)
        dev_data = self.build_valid_dataset(dev_data, batch_size)
        callbacks = self.build_callbacks(save_dir, **merge_dict(self.config, overwrite=True, logger=logger))
        # need to know #batches, otherwise progbar crashes
        dev_steps = math.ceil(self.num_samples_in(dev_data) / batch_size)
        checkpoint = get_callback_by_class(callbacks, tf.keras.callbacks.ModelCheckpoint)
        timer = Timer()
        try:
            history = self.train_loop(**merge_dict(self.config, trn_data=trn_data, dev_data=dev_data, epochs=epochs,
                                                   num_examples=num_examples,
                                                   train_steps_per_epoch=train_steps_per_epoch, dev_steps=dev_steps,
                                                   callbacks=callbacks, logger=logger, model=model, optimizer=optimizer,
                                                   loss=loss,
                                                   metrics=metrics, overwrite=True))
        except KeyboardInterrupt:
            print()
            if not checkpoint or checkpoint.best in (np.Inf, -np.Inf):
                self.save_weights(save_dir)
                logger.info('Aborted with model saved')
            else:
                logger.info(f'Aborted with model saved with best {checkpoint.monitor} = {checkpoint.best:.4f}')
            # noinspection PyTypeChecker
            history: tf.keras.callbacks.History() = get_callback_by_class(callbacks, tf.keras.callbacks.History)
        delta_time = timer.stop()
        best_epoch_ago = 0
        if history and hasattr(history, 'epoch'):
            trained_epoch = len(history.epoch)
            logger.info('Trained {} epochs in {}, each epoch takes {}'.
                        format(trained_epoch, delta_time, delta_time / trained_epoch if trained_epoch else delta_time))
            save_json(history.history, io_util.path_join(save_dir, 'history.json'), cls=io_util.NumpyEncoder)
            monitor_history: List = history.history.get(checkpoint.monitor, None)
            if monitor_history:
                best_epoch_ago = len(monitor_history) - monitor_history.index(checkpoint.best)
            if checkpoint and monitor_history and checkpoint.best != monitor_history[-1]:
                logger.info(f'Restored the best model saved with best '
                            f'{checkpoint.monitor} = {checkpoint.best:.4f} '
                            f'saved {best_epoch_ago} epochs ago')
                self.load_weights(save_dir)  # restore best model
        return history

    def train_loop(self, trn_data, dev_data, epochs, num_examples, train_steps_per_epoch, dev_steps, model, optimizer,
                   loss, metrics, callbacks,
                   logger, **kwargs):
        history = self.model.fit(trn_data, epochs=epochs, steps_per_epoch=train_steps_per_epoch,
                                 validation_data=dev_data,
                                 callbacks=callbacks,
                                 validation_steps=dev_steps,
                                 )  # type:tf.keras.callbacks.History
        return history

    def build_valid_dataset(self, dev_data, batch_size):
        dev_data = self.transform.file_to_dataset(dev_data, batch_size=batch_size, shuffle=False)
        return dev_data

    def build_train_dataset(self, trn_data, batch_size, num_examples):
        trn_data = self.transform.file_to_dataset(trn_data, batch_size=batch_size,
                                                  shuffle=True,
                                                  repeat=-1 if self.config.train_steps else None)
        return trn_data

    def build_callbacks(self, save_dir, logger, **kwargs):
        metrics = kwargs.get('metrics', 'accuracy')
        if isinstance(metrics, (list, tuple)):
            metrics = metrics[-1]
        monitor = f'val_{metrics}'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_dir, 'model.h5'),
            # verbose=1,
            monitor=monitor, save_best_only=True,
            mode='max',
            save_weights_only=True)
        logger.debug(f'Monitor {checkpoint.monitor} for checkpoint')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=io_util.makedirs(io_util.path_join(save_dir, 'logs')))
        csv_logger = FineCSVLogger(os.path.join(save_dir, 'train.log'), separator=' | ', append=True)
        callbacks = [checkpoint, tensorboard_callback, csv_logger]
        lr_decay_per_epoch = self.config.get('lr_decay_per_epoch', None)
        if lr_decay_per_epoch:
            learning_rate = self.model.optimizer.get_config().get('learning_rate', None)
            if not learning_rate:
                logger.warning('Learning rate decay not supported for optimizer={}'.format(repr(self.model.optimizer)))
            else:
                logger.debug(f'Created LearningRateScheduler with lr_decay_per_epoch={lr_decay_per_epoch}')
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: learning_rate / (1 + lr_decay_per_epoch * epoch)))
        anneal_factor = self.config.get('anneal_factor', None)
        if anneal_factor:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(factor=anneal_factor,
                                                                  patience=self.config.get('anneal_patience', 10)))
        early_stopping_patience = self.config.get('early_stopping_patience', None)
        if early_stopping_patience:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='max',
                                                              verbose=1,
                                                              patience=early_stopping_patience))
        return callbacks

    def on_train_begin(self):
        """
        Callback before the training starts
        """
        pass

    def predict(self, data: Any, batch_size=None, **kwargs):
        assert self.model, 'Please call fit or load before predict'
        if not data:
            return []
        data, flat = self.transform.input_to_inputs(data)

        if not batch_size:
            batch_size = self.config.batch_size

        dataset = self.transform.inputs_to_dataset(data, batch_size=batch_size, gold=kwargs.get('gold', False))

        results = []
        num_samples = 0
        data_is_list = isinstance(data, list)
        for idx, batch in enumerate(dataset):
            samples_in_batch = tf.shape(batch[-1] if isinstance(batch[-1], tf.Tensor) else batch[-1][0])[0]
            if data_is_list:
                inputs = data[num_samples:num_samples + samples_in_batch]
            else:
                inputs = None  # if data is a generator, it's usually one-time, not able to transform into a list
            for output in self.predict_batch(batch, inputs=inputs, **kwargs):
                results.append(output)
            num_samples += samples_in_batch

        if flat:
            return results[0]
        return results

    def predict_batch(self, batch, inputs=None, **kwargs):
        X = batch[0]
        Y = self.model.predict_on_batch(X)
        for output in self.transform.Y_to_outputs(Y, X=X, inputs=inputs, batch=batch, **kwargs):
            yield output

    @property
    def sample_data(self):
        return None

    @staticmethod
    def from_meta(meta: dict, **kwargs):
        """

        Parameters
        ----------
        meta
        kwargs

        Returns
        -------
        KerasComponent

        """
        cls = str_to_type(meta['class_path'])
        obj: KerasComponent = cls()
        assert 'load_path' in meta, f'{meta} doesn\'t contain load_path field'
        obj.load(meta['load_path'])
        return obj

    def export_model_for_serving(self, export_dir=None, version=1, overwrite=False, show_hint=False):
        assert self.model, 'You have to fit or load a model before exporting it'
        if not export_dir:
            assert 'load_path' in self.meta, 'When not specifying save_dir, load_path has to present'
            export_dir = get_resource(self.meta['load_path'])
        model_path = os.path.join(export_dir, str(version))
        if os.path.isdir(model_path) and not overwrite:
            logger.info(f'{model_path} exists, skip since overwrite = {overwrite}')
            return export_dir
        logger.info(f'Exporting to {export_dir} ...')
        tf.saved_model.save(self.model, model_path)
        logger.info(f'Successfully exported model to {export_dir}')
        if show_hint:
            logger.info(f'You can serve it through \n'
                        f'tensorflow_model_server --model_name={os.path.splitext(os.path.basename(self.meta["load_path"]))[0]} '
                        f'--model_base_path={export_dir} --rest_api_port=8888')
        return export_dir

    def serve(self, export_dir=None, grpc_port=8500, rest_api_port=0, overwrite=False, dry_run=False):
        export_dir = self.export_model_for_serving(export_dir, show_hint=False, overwrite=overwrite)
        if not dry_run:
            del self.model  # free memory
        logger.info('The inputs of exported model is shown below.')
        os.system(f'saved_model_cli show --all --dir {export_dir}/1')
        cmd = f'nohup tensorflow_model_server --model_name={os.path.splitext(os.path.basename(self.meta["load_path"]))[0]} ' \
              f'--model_base_path={export_dir} --port={grpc_port} --rest_api_port={rest_api_port} ' \
              f'>serve.log 2>&1 &'
        logger.info(f'Running ...\n{cmd}')
        if not dry_run:
            os.system(cmd)
