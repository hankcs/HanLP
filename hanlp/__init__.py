# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-13 18:05
import os

if not int(os.environ.get('HANLP_SHOW_TF_LOG', 0)):
    os.environ['VERBOSE'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
    import absl.logging, logging

    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.root.removeHandler(absl.logging._absl_handler)
    exec('absl.logging._warn_preinit_stderr = False')  # prevent exporting _warn_preinit_stderr

import hanlp.callbacks
import hanlp.common
import hanlp.components
import hanlp.datasets
import hanlp.layers
import hanlp.losses
import hanlp.metrics
import hanlp.optimizers
import hanlp.pretrained
import hanlp.utils

from hanlp.version import __version__

if not os.environ.get('HANLP_GREEDY_GPU', None):
    exec('from hanlp.utils.tf_util import nice_gpu')
    exec('nice_gpu()')

exec('''
from hanlp.utils.util import ls_resource_in_module
ls_resource_in_module(hanlp.pretrained)
''')


def load(save_dir: str, meta_filename='meta.json', transform_only=False, load_kwargs=None,
         **kwargs) -> hanlp.common.component.Component:
    """
    Load saved component from identifier.
    :param save_dir: The identifier to the saved component.
    :param meta_filename: The meta file of that saved component, which stores the class_path and version.
    :param transform_only: Whether to load transform only.
    :param load_kwargs: The arguments passed to `load`
    :param kwargs: Additional arguments parsed to the `from_meta` method.
    :return: A pretrained component.
    """
    save_dir = hanlp.pretrained.ALL.get(save_dir, save_dir)
    from hanlp.utils.component_util import load_from_meta_file
    return load_from_meta_file(save_dir, meta_filename, transform_only=transform_only, load_kwargs=load_kwargs, **kwargs)


def pipeline(*pipes) -> hanlp.components.pipeline.Pipeline:
    """
    Creates a pipeline of components.
    :param pipes: Components if pre-defined any.
    :return: A pipeline
    """
    return hanlp.components.pipeline.Pipeline(*pipes)
