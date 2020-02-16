# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-27 01:27
import logging
import os
import random
from typing import List

import numpy as np

from hanlp_common.constant import PAD


def set_gpu(idx=0):
    """Restrict TensorFlow to only use the GPU of idx

    Args:
      idx:  (Default value = 0)

    Returns:

    
    """
    gpus = get_visible_gpus()
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[idx], 'GPU')
            logical_devices = tf.config.experimental.list_logical_devices('GPU')
            assert len(logical_devices) == 1
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            # print(e)
            raise e


def get_visible_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    return gpus


def set_gpu_memory_growth(growth=True):
    gpus = get_visible_gpus()
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, growth)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            # print(e)
            raise e


def nice_gpu():
    """Use GPU nicely."""
    set_gpu_memory_growth()
    set_gpu()


def shut_up_python_logging():
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False


def set_tf_loglevel(level=logging.ERROR):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
    shut_up_python_logging()
    logging.getLogger('tensorflow').setLevel(level)


set_tf_loglevel()

shut_up_python_logging()
import tensorflow as tf

nice_gpu()


def size_of_dataset(dataset: tf.data.Dataset) -> int:
    count = 0
    for element in dataset.unbatch().batch(1):
        count += 1
    return count


def summary_of_model(model: tf.keras.Model):
    """https://stackoverflow.com/a/53668338/3730690

    Args:
      model: tf.keras.Model: 

    Returns:

    
    """
    if not model.built:
        return 'model structure unknown until calling fit() with some data'
    line_list = []
    model.summary(print_fn=lambda x: line_list.append(x))
    summary = "\n".join(line_list)
    return summary


def register_custom_cls(custom_cls, name=None):
    if not name:
        name = custom_cls.__name__
    tf.keras.utils.get_custom_objects()[name] = custom_cls


def set_seed_tf(seed=233):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def nice():
    nice_gpu()
    set_seed_tf()


def hanlp_register(arg):
    """Registers a class with the Keras serialization framework.

    Args:
      arg: 

    Returns:

    """
    class_name = arg.__name__
    registered_name = 'HanLP' + '>' + class_name

    # if tf_inspect.isclass(arg) and not hasattr(arg, 'get_config'):
    #     raise ValueError(
    #         'Cannot register a class that does not have a get_config() method.')

    tf.keras.utils.get_custom_objects()[registered_name] = arg

    return arg


def tensor_is_eager(tensor: tf.Tensor):
    return hasattr(tensor, 'numpy')


def copy_mask(src: tf.Tensor, dst: tf.Tensor):
    mask = getattr(src, '_keras_mask', None)
    if mask is not None:
        dst._keras_mask = mask
    return mask


def get_callback_by_class(callbacks: List[tf.keras.callbacks.Callback], cls) -> tf.keras.callbacks.Callback:
    for callback in callbacks:
        if isinstance(callback, cls):
            return callback


def tf_bernoulli(shape, p, dtype=None):
    return tf.keras.backend.random_binomial(shape, p, dtype)


def str_tensor_to_str(str_tensor: tf.Tensor) -> str:
    return str_tensor.numpy().decode('utf-8')


def str_tensor_2d_to_list(str_tensor: tf.Tensor, pad=PAD) -> List[List[str]]:
    l = []
    for i in str_tensor:
        sent = []
        for j in i:
            j = str_tensor_to_str(j)
            if j == pad:
                break
            sent.append(j)
        l.append(sent)
    return l


def str_tensor_to_list(pred):
    return [tag.predict('utf-8') for tag in pred]


def format_metrics(metrics: List[tf.keras.metrics.Metric]):
    return ' - '.join(f'{m.name}: {m.result():.4f}' for m in metrics)
