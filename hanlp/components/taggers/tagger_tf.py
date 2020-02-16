# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-25 21:49
import logging
from abc import ABC

import tensorflow as tf

from hanlp.common.keras_component import KerasComponent
from hanlp.layers.crf.crf_layer_tf import CRF, CRFLoss, CRFWrapper
from hanlp.metrics.chunking.iobes_tf import IOBES_F1_TF


class TaggerComponent(KerasComponent, ABC):

    def build_metrics(self, metrics, logger: logging.Logger, **kwargs):
        if metrics == 'f1':
            assert hasattr(self.transform, 'tag_vocab'), 'Name your tag vocab tag_vocab in your transform ' \
                                                         'or override build_metrics'
            if not self.config.get('run_eagerly', None):
                logger.debug('ChunkingF1 runs only under eager mode, '
                             'set run_eagerly=True to remove this warning')
            self.config.run_eagerly = True
            return IOBES_F1_TF(self.transform.tag_vocab)
        return super().build_metrics(metrics, logger, **kwargs)

    def build_loss(self, loss, **kwargs):
        assert self.model is not None, 'should create model before build loss'
        if loss == 'crf':
            if isinstance(self.model, tf.keras.models.Sequential):
                crf = CRF(len(self.transform.tag_vocab))
                self.model.add(crf)
                loss = CRFLoss(crf, self.model.dtype)
            else:
                self.model = CRFWrapper(self.model, len(self.transform.tag_vocab))
                loss = CRFLoss(self.model.crf, self.model.dtype)
            return loss
        return super().build_loss(loss, **kwargs)
