# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-27 14:22
import inspect
from abc import ABC, abstractmethod
from typing import Generator, Tuple, Union, Iterable, Any

import tensorflow as tf

from hanlp_common.structure import SerializableDict
from hanlp.common.vocab_tf import VocabTF
from hanlp.utils.io_util import get_resource
from hanlp.utils.log_util import logger


class Transform(ABC):

    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, **kwargs) -> None:
        super().__init__()
        self.map_y = map_y
        self.map_x = map_x
        if kwargs:
            if not config:
                config = SerializableDict()
            for k, v in kwargs.items():
                config[k] = v
        self.config = config
        self.output_types = None
        self.output_shapes = None
        self.padding_values = None

    @abstractmethod
    def fit(self, trn_path: str, **kwargs) -> int:
        """
        Build the vocabulary from training file

        Parameters
        ----------
        trn_path : path to training set
        kwargs

        Returns
        -------
        int
            How many samples in the training set
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def build_config(self):
        """
        By default, call build_types_shapes_values, usually called in component's build method.
        You can perform other building task here. Remember to call super().build_config
        """
        self.output_types, self.output_shapes, self.padding_values = self.create_types_shapes_values()
        # We prefer list over shape here, as it's easier to type [] than ()
        # if isinstance(self.output_shapes, tuple):
        #     self.output_shapes = list(self.output_shapes)
        # for i, shapes in enumerate(self.output_shapes):
        #     if isinstance(shapes, tuple):
        #         self.output_shapes[i] = list(shapes)
        #     for j, shape in enumerate(shapes):
        #         if isinstance(shape, tuple):
        #             shapes[j] = list(shape)

    @abstractmethod
    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Create dataset related values,
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abstractmethod
    def file_to_inputs(self, filepath: str, gold=True):
        """
        Transform file to inputs. The inputs are defined as raw features (e.g. words) to be processed into more
        features (e.g. forms and characters)

        Parameters
        ----------
        filepath
        gold
        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def inputs_to_samples(self, inputs, gold=False):
        if gold:
            yield from inputs
        else:
            for x in inputs:
                yield x, self.padding_values[-1]

    def file_to_samples(self, filepath: str, gold=True):
        """
        Transform file to samples
        Parameters
        ----------
        filepath
        gold
        """
        filepath = get_resource(filepath)
        inputs = self.file_to_inputs(filepath, gold)
        yield from self.inputs_to_samples(inputs, gold)

    def file_to_dataset(self, filepath: str, gold=True, map_x=None, map_y=None, batch_size=32, shuffle=None,
                        repeat=None,
                        drop_remainder=False,
                        prefetch=1,
                        cache=True,
                        **kwargs) -> tf.data.Dataset:
        """
        Transform file to dataset

        Parameters
        ----------
        filepath
        gold : bool
            Whether it's processing gold data or not. Example: there is usually a column for gold answer
            when gold = True.
        map_x : bool
            Whether call map_x or not. Default to self.map_x
        map_y : bool
            Whether call map_y or not. Default to self.map_y
        batch_size
        shuffle
        repeat
        prefetch
        kwargs

        Returns
        -------

        """

        # debug
        # for sample in self.file_to_samples(filepath):
        #     pass

        def generator():
            inputs = self.file_to_inputs(filepath, gold)
            samples = self.inputs_to_samples(inputs, gold)
            yield from samples

        return self.samples_to_dataset(generator, map_x, map_y, batch_size, shuffle, repeat, drop_remainder, prefetch,
                                       cache)

    def inputs_to_dataset(self, inputs, gold=False, map_x=None, map_y=None, batch_size=32, shuffle=None, repeat=None,
                          drop_remainder=False,
                          prefetch=1, cache=False, **kwargs) -> tf.data.Dataset:
        # debug
        # for sample in self.inputs_to_samples(inputs):
        #     pass

        def generator():
            samples = self.inputs_to_samples(inputs, gold)
            yield from samples

        return self.samples_to_dataset(generator, map_x, map_y, batch_size, shuffle, repeat, drop_remainder, prefetch,
                                       cache)

    def samples_to_dataset(self, samples: Generator, map_x=None, map_y=None, batch_size=32, shuffle=None, repeat=None,
                           drop_remainder=False,
                           prefetch=1, cache=True) -> tf.data.Dataset:
        output_types, output_shapes, padding_values = self.output_types, self.output_shapes, self.padding_values
        if not all(v for v in [output_shapes, output_shapes,
                               padding_values]):
            # print('Did you forget to call build_config() on your transform?')
            self.build_config()
            output_types, output_shapes, padding_values = self.output_types, self.output_shapes, self.padding_values
        assert all(v for v in [output_shapes, output_shapes,
                               padding_values]), 'Your create_types_shapes_values returns None, which is not allowed'
        # if not callable(samples):
        #     samples = Transform.generator_to_callable(samples)
        dataset = tf.data.Dataset.from_generator(samples, output_types=output_types, output_shapes=output_shapes)
        if cache:
            logger.debug('Dataset cache enabled')
            dataset = dataset.cache(cache if isinstance(cache, str) else '')
        if shuffle:
            if isinstance(shuffle, bool):
                shuffle = 1024
            dataset = dataset.shuffle(shuffle)
        if repeat:
            dataset = dataset.repeat(repeat)
        if batch_size:
            dataset = dataset.padded_batch(batch_size, output_shapes, padding_values, drop_remainder)
        if prefetch:
            dataset = dataset.prefetch(prefetch)
        if map_x is None:
            map_x = self.map_x
        if map_y is None:
            map_y = self.map_y
        if map_x or map_y:
            def mapper(X, Y):
                if map_x:
                    X = self.x_to_idx(X)
                if map_y:
                    Y = self.y_to_idx(Y)
                return X, Y

            dataset = dataset.map(mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    @abstractmethod
    def x_to_idx(self, x) -> Union[tf.Tensor, Tuple]:
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    @abstractmethod
    def y_to_idx(self, y) -> tf.Tensor:
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def lock_vocabs(self):
        for key, value in vars(self).items():
            if isinstance(value, VocabTF):
                value.lock()

    def summarize_vocabs(self, logger=None, header='Vocab summary:'):
        output = header + '\n'
        vocabs = {}
        for key, value in vars(self).items():
            if isinstance(value, VocabTF):
                vocabs[key] = value
        # tag vocab comes last usually
        for key, value in sorted(vocabs.items(), key=lambda kv: len(kv[1]), reverse=True):
            output += f'{key}' + value.summary(verbose=False) + '\n'
        output = output.strip()
        if logger:
            logger.info(output)
        else:
            print(output)

    @staticmethod
    def generator_to_callable(generator: Generator):
        return lambda: (x for x in generator)

    def str_to_idx(self, X, Y) -> Tuple[Union[tf.Tensor, Tuple], tf.Tensor]:
        return self.x_to_idx(X), self.y_to_idx(Y)

    def X_to_inputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]]) -> Iterable:
        return [repr(x) for x in X]

    def Y_to_outputs(self, Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False, inputs=None, X=None, batch=None) -> Iterable:
        return [repr(y) for y in Y]

    def XY_to_inputs_outputs(self, X: Union[tf.Tensor, Tuple[tf.Tensor]],
                             Y: Union[tf.Tensor, Tuple[tf.Tensor]], gold=False) -> Iterable:
        """
        Convert predicted tensors to outputs

        Parameters
        ----------
        X : Union[tf.Tensor, Tuple[tf.Tensor]]
            The inputs of model
        Y : Union[tf.Tensor, Tuple[tf.Tensor]]
            The outputs of model

        Returns
        -------

        """
        return [(x, y) for x, y in zip(self.X_to_inputs(X), self.Y_to_outputs(Y, gold))]

    def input_is_single_sample(self, input: Any) -> bool:
        return False

    def input_to_inputs(self, input: Any) -> Tuple[Any, bool]:
        """
        If input is one sample, convert it to a list which contains this unique sample

        Parameters
        ----------
        input :
            sample or samples

        Returns
        -------
        (inputs, converted) : Tuple[Any, bool]

        """
        flat = self.input_is_single_sample(input)
        if flat:
            input = [input]
        return input, flat

    def input_truth_output_to_str(self, input, truth, output):
        """
        Convert input truth output to string representation, usually for writing to file during evaluation

        Parameters
        ----------
        input
        truth
        output

        Returns
        -------

        """
        return '\t'.join([input, truth, output]) + '\n'
