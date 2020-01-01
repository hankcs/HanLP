# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-06-13 18:05

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

import os

if not os.environ.get('HANLP_SHOW_TF_LOG', None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    import absl.logging, logging

    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.root.removeHandler(absl.logging._absl_handler)
    exec('absl.logging._warn_preinit_stderr = False')  # prevent exporting _warn_preinit_stderr

if not os.environ.get('HANLP_GREEDY_GPU', None):
    exec('from hanlp.utils.tf_util import nice_gpu')
    exec('nice_gpu()')

exec('''
from hanlp.utils.util import ls_resource_in_module
ls_resource_in_module(hanlp.pretrained)
''')


def load(save_dir, meta_filename='meta.json', **kwargs) -> hanlp.common.component.Component:
    save_dir = hanlp.pretrained.ALL.get(save_dir, save_dir)
    from hanlp.utils.component_util import load_from_meta_file
    return load_from_meta_file(save_dir, meta_filename, **kwargs)


def pipeline(*pipes) -> hanlp.components.pipeline.Pipeline:
    return hanlp.components.pipeline.Pipeline(*pipes)
